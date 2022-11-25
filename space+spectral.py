'''
对空间信息进行掩码信息提取。
'''
import argparse
import datetime
import os
import random
import time
import copy

import numpy as np
import torch
from torch.nn import functional as F
from timm.utils import AverageMeter
from torch.autograd import Variable
from torch.backends import cudnn
from torch import optim, nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from config import get_config
from data import get_hsi_spectral_dataloader, get_virtual_dataloader, get_mask_dataloader, get_hsi_spatial_dataloader, \
    get_hsi_spatial_spectral_dataloader
from data.utils import get_tensor_dataset
from eval_method import get_eval_method
from logger import create_logger
from lr_scheduler import build_scheduler, build_finetune_scheduler
from model import get_pretrain_model, get_finetune_G, get_G, get_decoder, get_mix_model
from model.Trans_BCDM_A.net_A import ResClassifier
from model.Trans_BCDM_A.utils_A import cdd
from optimizer import build_optimizer, build_optimizer_c


from utils import get_grad_norm, save_checkpoint

def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # 定义是否分布式
    parser.add_argument('--is_dist', default=False, type=bool, help="is distrubution")
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-source-path', type=str, help='path to source dataset')
    parser.add_argument('--label-source-path', type=str, help='path to source label')
    parser.add_argument('--data-target-path', type=str, help='path to target dataset')
    parser.add_argument('--label-target-path', type=str, help='path to target label')
    # 继续？继续什么呢
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    # checkpoint pytorch 推出的一个节省缓存的功能
    # action store_true 当在命令行中不指定时 为默认值。 如果加入了 use-checkpoint 不指定值就可以设置为true。粗浅说明
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    # apex 的参数 混合精度加速 choices是对应函数的参数
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    # 输出文件的根目录
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    # 实验的tag
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    # eta
    parser.add_argument("--eta", type=float, required=True, help='eta of entropy')
    # mask_ratio
    parser.add_argument("--mask-ratio", type=float, required=True, help='mask ratio')
    # refactor_eta
    parser.add_argument("--refactor-eta", type=float, required=True, help='eta of refactor loss')
    # depth
    parser.add_argument("--attention-depth", type=int, required=True, help='eta of refactor loss')
    # 解析参数
    args = parser.parse_args()
    # 得到yacs cfgNOde，值是原有的值
    config = get_config(args)

    return args, config




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(config):
    assert config.DATA.MODE == 'spatial+spectral','mode is not spatial'
    # 获取数据
    if on_mac:
        data_loader_train = get_mask_dataloader(config, size=(128, 48, 5, 5))
        # 微调的网络
        test_loader = get_mask_dataloader(config, size=(128, 48, 5, 5))
        train_src_loader = get_mask_dataloader(config, size=(128, 48, 5, 5))
        train_tgt_loader = get_mask_dataloader(config, size=(128, 48, 5, 5))
    else:
        assert config.DATA.MODE in ['spectral', 'spatial+spectral', 'spatial'],f'this mode:{config.DATA.MODE} not support yet'
        if config.DATA.MODE == 'spectral':
            test_loader, train_src_loader, train_tgt_loader = get_hsi_spectral_dataloader(config)
        elif config.DATA.MODE == 'spatial':
            test_loader, train_src_loader, train_tgt_loader = get_hsi_spatial_dataloader(config)
        elif config.DATA.MODE == 'spatial+spectral':
            test_loader, train_src_loader, train_tgt_loader = get_hsi_spatial_spectral_dataloader(config)
        else:
            raise Exception(f'mode {config.DATA.MODE} not support')

        # finetune_test_loader, finetune_train_src_loader, finetune_train_tgt_loader = get_hsi_spatial_dataloader(config)
    # 设置模型及优化器，不设置动态更新学习率了
    device = 'cpu' if on_mac else 'cuda'
    finetune_epochs = 20
    # device = 'cpu'
    # xian kongjian hou guangpu
    # 1 write a mix model mix two decoder output
    # 2 change the way
    G_spatial,G_spectral = get_G(config)
    G_spatial = G_spatial.to(device)
    G_spectral = G_spectral.to(device)
    Decoder_spatial, Decoder_spectral = get_decoder(config)
    Decoder_spatial = Decoder_spatial.to(device)
    Decoder_spectral = Decoder_spectral.to(device)
    mix_model = get_mix_model(config).to(device)
    C1 = ResClassifier(num_classes=config.DATA.CLASS_NUM, num_unit=config.MODEL.CLASSIFIER_IN_DIM).to(device)
    C2 = ResClassifier(num_classes=config.DATA.CLASS_NUM, num_unit=config.MODEL.CLASSIFIER_IN_DIM).to(device)
    logger.info(f'G_spatial:{str(G_spatial)}')
    logger.info(f'G_spectral:{str(G_spectral)}')
    logger.info(f'decoder_spatial:{str(Decoder_spatial)}')
    logger.info(f'decoder_spectral:{str(Decoder_spectral)}')
    logger.info(f'mix model:{str(mix_model)}')
    logger.info(f'C1:{str(C1)}')
    logger.info(f'C2:{str(C2)}')
    # optimizer
    G_spectral_optimizer = build_optimizer(config, G_spectral, logger, is_pretrain=True)
    G_spatial_optimizer = build_optimizer(config, G_spatial, logger, is_pretrain=True)
    Decoder_spectral_optimizer = build_optimizer(config, Decoder_spectral, logger, is_pretrain=True)
    Decoder_spatial_optimizer = build_optimizer(config, Decoder_spatial, logger, is_pretrain=True)
    mix_model_optimizer = build_optimizer(config, mix_model, logger, is_pretrain=True)
    C_optimizer = build_optimizer_c(C1, C2, config, logger)
    # scheduler
    sche_length = min(len(train_src_loader), len(train_tgt_loader))
    G_spatial_scheduler = build_scheduler(config, G_spatial_optimizer, sche_length)
    G_spectral_scheduler = build_scheduler(config, G_spectral_optimizer, sche_length)
    Decoder_spectral_scheduler = build_scheduler(config, Decoder_spectral_optimizer, sche_length)
    Decoder_spatial_scheduler = build_scheduler(config, Decoder_spatial_optimizer, sche_length)
    mix_model_scheduler = build_scheduler(config, mix_model_optimizer, sche_length)
    C_scheduler = build_scheduler(config, C_optimizer, sche_length)

    logger.info("Start training")
    start_time = time.time()
    # # 0的数组，size：10 1
    # acc = np.zeros([nDataSet, 1])
    # #
    # A = np.zeros([nDataSet, CLASS_NUM])
    # K = np.zeros([nDataSet, 1])
    for epoch in range(config.TRAIN.EPOCHS):
        # train
        train_one_epoch(config=config,
                        G_spatial=G_spatial,
                        G_spectral=G_spectral,
                        Decoder_spatial=Decoder_spatial,
                        Decoder_spectral=Decoder_spectral,
                        mix_model=mix_model,
                        C1=C1, C2=C2,
                        src_train_loader=train_src_loader,
                        tgt_train_loader=train_tgt_loader,
                        G_spatial_optim=G_spatial_optimizer,
                        G_spectral_optim=G_spectral_optimizer,
                        Decoder_spatial_optim=Decoder_spatial_optimizer,
                        Decoder_spectral_optim=Decoder_spectral_optimizer,
                        mix_model_optim=mix_model_optimizer,
                        C_optim=C_optimizer,
                        G_spatial_scheduler=G_spatial_scheduler,
                        G_spectral_schduler=G_spectral_scheduler,
                        Decoder_spatial_scheduler=Decoder_spatial_scheduler,
                        Decoder_spectral_scheduler=Decoder_spectral_scheduler,
                        mix_model_schduler=mix_model_scheduler,
                        C_scheduler=C_scheduler, epoch=epoch
                        )
        # # pretrain
        # pretrain_train_one_epoch(config, pretrain_model, data_loader_train, optimizer_pretrain, epoch)
        # # finetune
        # finetune_train_one_epoch(config, pretrain_model, finetune_model, C1, C2, finetune_train_src_loader,
        #                          finetune_train_tgt_loader, optimizer_finetune, optimizer_C, epoch)
        # # eval_one_epoch(config, pretrain_model, finetune_model, C1, C2, finetune_test_loader)
        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            # eval on train dataset
            eval_train(config=config, finetune_train_src_loader=train_src_loader,
                       finetune_train_tgt_loader=train_tgt_loader, G_spatial=G_spatial,
                       G_spectral=G_spectral, mix_model=mix_model, C1=C1, C2=C2, epoch=epoch)
            # eval on test dataset
            eval_one_epoch(config=config, G_spatial=G_spatial,G_spectral=G_spectral,mix_model=mix_model,
                           C1=C1, C2=C2, test_loader=test_loader,epoch=epoch)

            # save model
            # save_checkpoint(config, epoch, pretrain_model, finetune_model, C1, C2, 0., pretrain_optimizer,
            #                 finetune_optimizer, C_optimizer, logger)


def train_one_epoch(config, G_spatial, G_spectral, Decoder_spatial, Decoder_spectral,
                    mix_model, C1, C2, src_train_loader,tgt_train_loader,
                    G_spatial_optim, G_spectral_optim, Decoder_spatial_optim, Decoder_spectral_optim,
                    mix_model_optim, C_optim, G_spatial_scheduler, G_spectral_schduler,
                    Decoder_spatial_scheduler,Decoder_spectral_scheduler,mix_model_schduler,
                    C_scheduler, epoch):
    # 需要更换样本吗
    if on_mac:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    eta = config.TRAIN.ETA
    rf_eta = config.TRAIN.RF_ETA
    G_spatial.train()
    G_spectral.train()
    Decoder_spatial.train()
    Decoder_spectral.train()
    mix_model.train()
    C1.train()
    C2.train()
    finetune_step = min(len(src_train_loader), len(tgt_train_loader))

    train_pred_all = []
    train_all = []
    correct = 0
    total = 0

    time_start_per_epoch = time.time()
    # shuju jiazai meixiehao
    for batch_idx, data in enumerate(zip(src_train_loader, tgt_train_loader)):
        # (data_s, mask_s, label_s), (data_t, mask_t, label_t) = data
        (s_spatial, s_spatial_mask, s_spectral, s_spectral_mask, s_label) = data[0]
        (t_spatial, t_spatial_mask, t_spectral, t_spectral_mask, t_label) = data[1]
        # assert data_s.shape == data_t.shape, '数据形状不一致'
        data_loader = [[], []]
        for s, t in [[s_spatial, t_spatial], [s_spatial_mask, t_spatial_mask], [s_spectral,t_spectral],
                     [s_spectral_mask, t_spectral_mask], [s_label,t_label]]:
            s_1, s_2 = s.chunk(2, 0)
            t_1, t_2 = t.chunk(2, 0)
            data_loader[0].append(Variable(torch.cat((s_1, t_2))))
            data_loader[1].append(Variable(torch.cat((t_1, s_2))))

        if not on_mac:
            s_spatial, s_spatial_mask, s_spectral, s_spectral_mask, s_label = s_spatial.cuda(), s_spatial_mask.cuda(), s_spectral.cuda(), s_spectral_mask.cuda(), s_label.cuda()
            t_spatial, t_spatial_mask, t_spectral, t_spectral_mask, t_label = t_spatial.cuda(), t_spatial_mask.cuda(), t_spectral.cuda(), t_spectral_mask.cuda(), t_label.cuda()
        spatial_all = Variable(torch.cat((s_spatial, t_spatial), 0))
        spectral_all = Variable(torch.cat((s_spectral, t_spectral), 0))
        # data_all = data_all.type(torch.LongTensor)
        s_label = s_label.long()
        t_lebel = t_label.long()
        s_label = Variable(s_label)
        t_label = Variable(t_lebel)
        bs = len(s_label)

        """source domain discriminative"""
        # Step A train all networks to minimize loss on source
        G_spatial_optim.zero_grad()
        G_spectral_optim.zero_grad()
        mix_model_optim.zero_grad()
        C_optim.zero_grad()
        # temp_d = copy.deepcopy(pretrain_model.state_dict())
        output_spatial = G_spatial(spatial_all, mask=None)
        output_spectral = G_spectral(spectral_all,mask=None)
        output = mix_model(output_spatial, output_spectral)
        # 输出size是64 1024
        output1 = C1(output)
        output2 = C2(output)
        output_s1 = output1[:bs, :]
        output_s2 = output2[:bs, :]
        output_t1 = output1[bs:, :]
        output_t2 = output2[bs:, :]
        output_t1 = F.softmax(output_t1, dim=1)
        output_t2 = F.softmax(output_t2, dim=1)
        entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
        loss1 = criterion(output_s1, s_label)
        loss2 = criterion(output_s2, t_label)

        # all_loss = loss1 + loss2 + 0.01 * entropy_loss
        all_loss = loss1 + loss2
        all_loss.backward()
        G_spatial_optim.step()
        G_spectral_optim.step()
        mix_model_optim.step()
        C_optim.step()
        writer.add_scalar(tag='stepA_loss1',scalar_value=loss1,global_step=epoch*finetune_step+batch_idx)
        writer.add_scalar(tag='stepA_loss2', scalar_value=loss2,global_step=epoch*finetune_step+batch_idx)
        # writer.add_scalar(tag='stepA_entropy', scalar_value=entropy_loss,global_step=epoch*finetune_step+batch_idx)
        writer.add_scalar(tag='stepA_all(0.01entropy)', scalar_value=all_loss,global_step=epoch*finetune_step+batch_idx)
        logger.info(f'stepA_loss1:{loss1}\tstepA_loss2:{loss2}\tstepA_entropy:{entropy_loss}\tstepA_all(0.01entropy):{all_loss}')

        """target domain discriminative"""
        # Step B train classifier to maximize discrepancy
        G_spatial_optim.zero_grad()
        G_spectral_optim.zero_grad()
        mix_model_optim.zero_grad()
        C_optim.zero_grad()

        output_spatial = G_spatial(spatial_all, mask=None)
        output_spectral = G_spectral(spectral_all, mask=None)
        output = mix_model(output_spatial, output_spectral)
        output1 = C1(output)
        output2 = C2(output)
        output_s1 = output1[:bs, :]
        output_s2 = output2[:bs, :]
        output_t1 = output1[bs:, :]
        output_t2 = output2[bs:, :]
        output_t1 = F.softmax(output_t1, dim=1)
        output_t2 = F.softmax(output_t2, dim=1)

        loss1 = criterion(output_s1, s_label)
        loss2 = criterion(output_s2, s_label)
        entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
        loss_dis = cdd(output_t1, output_t2)
        # F_loss = loss1 + loss2 - eta * loss_dis + 0.01 * entropy_loss
        F_loss = loss1 + loss2 - eta * loss_dis
        writer.add_scalar(tag='stepB_loss1', scalar_value=loss1,global_step=epoch*finetune_step+batch_idx)
        writer.add_scalar(tag='stepB_loss2', scalar_value=loss2,global_step=epoch*finetune_step+batch_idx)
        # writer.add_scalar(tag='stepB_entropy', scalar_value=entropy_loss,global_step=epoch*finetune_step+batch_idx)
        writer.add_scalar(tag='stepB_cdd', scalar_value=loss_dis,global_step=epoch*finetune_step+batch_idx)
        writer.add_scalar(tag='stepB_all', scalar_value=F_loss,global_step=epoch*finetune_step+batch_idx)
        logger.info(f'stepB_loss1:{loss1}\tstepB_loss2:{loss2}\tstepB_entropy:{entropy_loss}\tstepB_cdd:{loss_dis}\tstepB_all:{F_loss}')
        F_loss.backward()
        C_optim.step()

        # Step C train genrator to minimize discrepancy

        '''stepc train genrator to minimize discrepancy'''
        NUM_K = 4
        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()

        start = time.time()
        end = time.time()
        for i in range(NUM_K):
            ''' refactor'''
            G_spectral_optim.zero_grad()
            G_spatial_optim.zero_grad()
            Decoder_spatial_optim.zero_grad()
            Decoder_spectral_optim.zero_grad()
            mix_model_optim.zero_grad()
            # index_count = 0
            spatial_refactor_loss = 0.0
            spectral_refactor_loss = 0.0
            for idx, (refactor_spatial,refactor_spatial_mask,refactor_spectral,refactor_spectral_mask, _) in enumerate(data_loader):
                # index_count += 1
                # non-blocking 不会堵塞与其无关的的事情
                # img size 128 192 192
                # mask size 128 48 48
                # 遮盖比率为0.75
                if not on_mac:
                    refactor_spatial = refactor_spatial.cuda(non_blocking=True)
                    refactor_spatial_mask = refactor_spatial_mask.cuda(non_blocking=True)
                    refactor_spectral = refactor_spectral.cuda(non_blocking=True)
                    refactor_spectral_mask = refactor_spectral_mask.cuda(non_blocking=True)
                # 从模型的结果得到一个loss
                spatial_feature = G_spatial(refactor_spatial, refactor_spatial_mask)
                spectral_feature = G_spectral(refactor_spectral,refactor_spectral_mask)
                spatial_refactor_loss += Decoder_spatial(x=refactor_spatial, mask=refactor_spatial_mask,
                                                         rec=spatial_feature)
                spectral_refactor_loss += Decoder_spectral(x=refactor_spectral, mask=refactor_spectral_mask,
                                                         rec=spectral_feature)
                # 更新参数
                # loss_refactor.backward()
                # if config.TRAIN.CLIP_GRAD:
                #     grad_norm = torch.nn.utils.clip_grad_norm_(G.parameters(), config.TRAIN.CLIP_GRAD)
                # else:
                #     grad_norm = get_grad_norm(G.parameters())
                # # pretrain_optim.step()
                # # lr_scheduler.step_update(epoch * num_steps + idx)
                # if not on_mac:
                #     torch.cuda.synchronize()
                #
                # loss_meter.update(refactor_loss.item(), img.size(0))
                # norm_meter.update(grad_norm)
                # batch_time.update(time.time() - end)
                end = time.time()



            # pretrain_scheduler.step_update(epoch * finetune_step * 2 + batch_idx * 2 + idx)
            # epoch_time = time.time() - start
            # logger.info(f"INDEX_COUNT {epoch} index_count is {index_count}")
            # logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
            # G.zero_grad()
            C1.train()
            C2.train()
            C_optim.zero_grad()

            output_spatial = G_spatial(spatial_all, mask=None)
            output_spectral = G_spectral(spectral_all, mask=None)
            output = mix_model(output_spatial, output_spectral)
            features_source = output[:bs, :]
            features_target = output[bs:, :]
            output1 = C1(output)
            output2 = C2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1 = F.softmax(output_t1, dim=1)
            output_t2 = F.softmax(output_t2, dim=1)

            entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
            loss_dis = cdd(output_t1, output_t2)
            # D_loss = eta * loss_dis + 0.01 * entropy_loss
            D_loss = eta * loss_dis
            step_3_all_loss = D_loss + rf_eta * spectral_refactor_loss + rf_eta * spatial_refactor_loss
            writer.add_scalar(tag='stepC_CDD_loss', scalar_value=loss_dis, global_step=epoch * finetune_step*NUM_K + batch_idx*NUM_K+i)
            # writer.add_scalar(tag='stepC_D_loss', scalar_value=D_loss,
            #                   global_step=epoch * finetune_step + batch_idx * NUM_K + i)
            writer.add_scalar(tag='stepC_spectral_refactor_loss', scalar_value=rf_eta * spectral_refactor_loss,
                              global_step=epoch * finetune_step*NUM_K + batch_idx*NUM_K+i)
            writer.add_scalar(tag='stepC_spatial_refactor_loss', scalar_value=rf_eta * spatial_refactor_loss,
                              global_step=epoch * finetune_step * NUM_K + batch_idx * NUM_K + i)
            writer.add_scalar(tag='stepC_all_loss', scalar_value=step_3_all_loss,
                              global_step=epoch * finetune_step*NUM_K + batch_idx*NUM_K+i)
            logger.info(f'stepC_D_loss:{D_loss}\tstepC_spectral_refactor_loss:{rf_eta * spectral_refactor_loss}\t'
                        f'stepC_spatial_refactor_loss:{rf_eta * spatial_refactor_loss}\tstepC_all_loss:{step_3_all_loss}')
            step_3_all_loss.backward()
            G_spatial_optim.step()
            G_spectral_optim.step()
            Decoder_spectral_optim.step()
            Decoder_spatial_optim.step()
            mix_model_optim.step()
            C_optim.step()
        lr = G_spatial_optim.param_groups[0]['lr']
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        # etas = batch_time.avg * (num_steps - idx)
        logger.info(
            f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{batch_idx}/{finetune_step}]\t'
            f'lr {lr:.6f}\t'
            f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
            f'mem {memory_used:.0f}MB')
        # D_loss.backward()
        # scheduler step up
        G_spatial_scheduler.step_update(epoch * finetune_step + batch_idx)
        G_spectral_schduler.step_update(epoch * finetune_step + batch_idx)
        Decoder_spatial_scheduler.step_update(epoch * finetune_step + batch_idx)
        Decoder_spectral_scheduler.step_update(epoch * finetune_step + batch_idx)
        mix_model_schduler.step_update(epoch * finetune_step + batch_idx)
        C_scheduler.step_update(epoch * finetune_step + batch_idx)
    logger.info('Train Ep: {} \tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f} '.format(
        epoch, loss1.item(), loss2.item(), loss_dis.item(), entropy_loss.item()))
    time_end_per_epoch = time.time()
    logger.info(f'time_{epoch}_epoch:{(time_end_per_epoch - time_start_per_epoch)}')





def eval_train(config, finetune_train_src_loader, finetune_train_tgt_loader, G_spatial, G_spectral, mix_model, C1, C2, epoch):
    G_spatial.eval()
    G_spectral.eval()
    mix_model.eval()
    C1.eval()
    C2.eval()
    val_pred_all = []
    val_all = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(zip(finetune_train_src_loader, finetune_train_tgt_loader)):
            (s_spatial, s_spatial_mask, s_spectral, s_spectral_mask, s_label) = data[0]
            (t_spatial, t_spatial_mask, t_spectral, t_spectral_mask, t_label) = data[1]
            if not on_mac:
                s_spatial, s_spatial_mask, s_spectral, s_spectral_mask, s_label = s_spatial.cuda(), s_spatial_mask.cuda(), s_spectral.cuda(), s_spectral_mask.cuda(), s_label.cuda()
                t_spatial, t_spatial_mask, t_spectral, t_spectral_mask, t_label = t_spatial.cuda(), t_spatial_mask.cuda(), t_spectral.cuda(), t_spectral_mask.cuda(), t_label.cuda()

            s_spatial_feature = G_spatial(s_spatial, mask=None)
            s_spectral_feature = G_spectral(s_spectral, mask=None)
            output_s = mix_model(s_spatial_feature, s_spectral_feature)
            t_spatial_feature = G_spatial(t_spatial, mask=None)
            t_spectral_feature = G_spectral(t_spectral, mask=None)
            output_t = mix_model(t_spatial_feature, t_spectral_feature)
            output_s_1 = C1(output_s)
            output_s_2 = C2(output_s)
            output_t_1 = C1(output_t)
            output_t_2 = C2(output_t)
            output_s_add = output_s_1 + output_s_2
            output_t_add = output_t_1 + output_t_2
            _, predicted_s = torch.max(output_s_add.data, 1)
            _, predicted_t = torch.max(output_t_add.data, 1)
            total += s_label.size(0)+t_label.size(0)
            val_all = np.concatenate([val_all, s_label.data.cpu().numpy(),t_label.data.cpu().numpy()])
            val_pred_all = np.concatenate([val_pred_all, predicted_s.cpu().numpy(),predicted_t.cpu().numpy()])
            correct += predicted_s.eq(s_label.data.view_as(predicted_s)).cpu().sum().item()
            correct += predicted_t.eq(t_label.data.view_as(predicted_t)).cpu().sum().item()
        test_accuracy = 100. * correct / total

        # acc[iDataSet] = test_accuracy
        # # OA = acc
        # C = metrics.confusion_matrix(val_all, val_pred_all)
        # A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
        # K[iDataSet] = metrics.cohen_kappa_score(val_all, val_pred_all)
        writer.add_scalar(tag='train_acc',scalar_value=100. * correct / total,global_step=epoch)
        logger.info('\ttrain dataset Accuracy: {}/{} ({:.2f}%)\t'.format(correct, total, 100. * correct / total))

def eval_one_epoch(config, G_spatial, G_spectral, mix_model, C1, C2, test_loader, epoch):
    G_spatial.eval()
    G_spectral.eval()
    mix_model.eval()
    C1.eval()
    C2.eval()
    val_pred_all = []
    val_all = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (spatial,spatial_mask,spectral,spectral_mask, _)  in enumerate(test_loader):
            if not on_mac:
                spatial = spatial.cuda(non_blocking=True)
                spatial_mask = spatial_mask.cuda(non_blocking=True)
                spectral = spectral.cuda(non_blocking=True)
                spectral_mask = spectral_mask.cuda(non_blocking=True)
                label = _.cuda(non_blocking=True)

            spatial_feature = G_spatial(spatial,mask=None)
            spectral_feature = G_spectral(spectral,mask=None)
            output = mix_model(spatial_feature,spectral_feature)
            output1 = C1(output)
            output2 = C2(output)
            output_add = output1 + output2
            _, predicted = torch.max(output_add.data, 1)
            total += label.size(0)
            val_all = np.concatenate([val_all, label.data.cpu().numpy()])
            val_pred_all = np.concatenate([val_pred_all, predicted.cpu().numpy()])
            correct += predicted.eq(label.data.view_as(predicted)).cpu().sum().item()
        test_accuracy = 100. * correct / total
        eval_method.set_value(test_accuracy,val_all,val_pred_all)
        writer.add_scalar(tag='test_acc', scalar_value=100. * correct / total,global_step=epoch)
        logger.info('\tval_Accuracy: {}/{} ({:.2f}%)\t'.format(correct, total, 100. * correct / total))


if __name__ == '__main__':
    _, config = parse_option()
    on_mac = False if torch.cuda.is_available() else False


    if not config.IS_DIST:
        os.environ['RANK'] = '-1'
        os.environ['world_size'] = '-1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1080'

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    if on_mac:
        torch.cuda.set_device(config.LOCAL_RANK)
    else:
        torch.cuda.set_device(config.LOCAL_RANK)
    if on_mac:
        torch.cuda.set_device(config.LOCAL_RANK)
    else:
        torch.cuda.set_device(config.LOCAL_RANK)
    # seed = config.SEED
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    cudnn.benchmark = True
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # # gradient accumulation also need to scale the learning rate
    # if config.TRAIN.ACCUMULATION_STEPS > 1:
    #     linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS

    # 先是可变参数，然后变完参数后冻结 上面的没太看懂，去除dist之后查看main
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")
    # 估摸着也就是看看是不是主机
    if 1:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    seeds = [1330, 1220, 1336, 1337, 1334, 1236, 1226, 1235, 1228, 1229]
    eval_method = get_eval_method(seeds,config.DATA.CLASS_NUM)
    for i in range(len(seeds)):
        eval_method.set_seed(seeds[i])
        writer = SummaryWriter(log_dir=config.OUTPUT+'/'+config.TAG+'_seed'+str(seeds[i]))
        logger.info(f'seed:{seeds[i]}')
        seed_everything(seeds[i])
        main(config)
    eval_method.get_OA_AA_KAPPA(logger)
