'''
首先是预训练的时候，每个batch里面尽量均匀的分布着目标域与源域的数据。两域的样本总数量不一样，每类样本数量不一样。
初步的设想是：
1，每个batch里面，均匀包含源域和目标域，均匀包含每个类，每个类的样本数目相同。
2，编码器与微调的模型脱离
3，数据的问题，数据需不需要做分离？在大数据上做预训练，再在小的数据上进行微调？
4，特征提取阶段，mae通过卷积进行的特征提取，可以试着写一个单独的mlp的写法
5，注意力模型，师哥的dtransformer也使用了cls_token，也可以使用vit的结构进行实验
decoder + decoder:预训练 -> decoder+attention+2 Resclassifier : 微调，step A，B，C。decoder不更新参数
'''
import argparse
import datetime
import os
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

from config import get_config
from data import get_hsi_spectral_dataloader, get_virtual_dataloader, get_mask_dataloader
from data.utils import get_tensor_dataset
from logger import create_logger
from lr_scheduler import build_scheduler, build_finetune_scheduler
from model import get_pretrain_model, get_finetune_G
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
    # 解析参数
    args = parser.parse_args()
    # 得到yacs cfgNOde，值是原有的值
    config = get_config(args)

    return args, config


def main(config):
    # 获取数据
    if on_mac:
        data_loader_train = get_mask_dataloader(config, size=(128, 48, 5, 5))
        # 微调的网络
        finetune_test_loader = get_mask_dataloader(config, size=(128, 48, 5, 5))
        finetune_train_src_loader = get_mask_dataloader(config, size=(128, 48, 5, 5))
        finetune_train_tgt_loader = get_mask_dataloader(config, size=(128, 48, 5, 5))
    else:
        finetune_test_loader, finetune_train_src_loader, finetune_train_tgt_loader = get_hsi_spectral_dataloader(
            config, is_pretrain=True)
    # 设置模型及优化器，不设置动态更新学习率了
    device = 'cpu' if on_mac else 'cuda'
    finetune_epochs = 20
    # device = 'cpu'
    pretrain_model = get_pretrain_model(config).to(device=device)
    finetune_model = get_finetune_G(config).to(device=device)
    C1 = ResClassifier(num_classes=config.DATA.CLASS_NUM, num_unit=512).to(device)
    C2 = ResClassifier(num_classes=config.DATA.CLASS_NUM, num_unit=512).to(device)
    # optimizer
    pretrain_optimizer = build_optimizer(config, pretrain_model, logger, is_pretrain=True)
    finetune_optimizer = build_optimizer(config, finetune_model, logger, is_pretrain=False)
    C_optimizer = build_optimizer_c(C1, C2, config, logger)
    # scheduler
    sche_length = min(len(finetune_train_src_loader), len(finetune_train_tgt_loader))
    pretrain_scheduler = build_scheduler(config, pretrain_optimizer, sche_length * 2)
    finetune_scheduler = build_scheduler(config, finetune_optimizer, sche_length)
    C_scheduler = build_scheduler(config, C_optimizer, sche_length)
    logger.info("Start training")
    start_time = time.time()
    # # 0的数组，size：10 1
    # acc = np.zeros([nDataSet, 1])
    # #
    # A = np.zeros([nDataSet, CLASS_NUM])
    # K = np.zeros([nDataSet, 1])

    seeds = [1330, 1220, 1336, 1337, 1334, 1236, 1226, 1235, 1228, 1229]
    for epoch in range(config.TRAIN.EPOCHS):
        # train
        train_one_epoch(config, pretrain_model, finetune_model, C1, C2, finetune_train_src_loader,
                        finetune_train_tgt_loader, pretrain_optimizer, finetune_optimizer, C_optimizer,
                        pretrain_scheduler, finetune_scheduler, C_scheduler, epoch)
        # # pretrain
        # pretrain_train_one_epoch(config, pretrain_model, data_loader_train, optimizer_pretrain, epoch)
        # # finetune
        # finetune_train_one_epoch(config, pretrain_model, finetune_model, C1, C2, finetune_train_src_loader,
        #                          finetune_train_tgt_loader, optimizer_finetune, optimizer_C, epoch)
        # # eval_one_epoch(config, pretrain_model, finetune_model, C1, C2, finetune_test_loader)
        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            temp_finetune_optimizer = build_optimizer(config, finetune_model, logger, is_pretrain=False)
            temp_C_optimizer = build_optimizer_c(C1, C2, config, logger)
            temp_sche_length = min(len(finetune_train_src_loader), len(finetune_train_tgt_loader))
            temp_finetune_scheduler = build_finetune_scheduler(config, temp_finetune_optimizer, temp_sche_length,
                                                          ft_epochs=finetune_epochs)
            temp_C_scheduler = build_finetune_scheduler(config, temp_C_optimizer, temp_sche_length, ft_epochs=finetune_epochs)
            # 复制一下原始双分类器的值
            E_state = copy.deepcopy(finetune_model.state_dict())
            C1_state = copy.deepcopy(C1.state_dict())
            C2_state = copy.deepcopy(C2.state_dict())
            # eval
            eval_one_epoch(config, pretrain_model, finetune_model, C1, C2, finetune_test_loader)
            # finetune
            finetune_train_epochs(config,pretrain_model,finetune_model,C1,C2,finetune_train_src_loader,
                                  finetune_train_tgt_loader,temp_finetune_optimizer,temp_C_optimizer,
                                  temp_finetune_scheduler,temp_C_scheduler,
                                  epoch,finetune_epochs)
            # eval
            eval_one_epoch(config, pretrain_model, finetune_model, C1, C2, finetune_test_loader)

            # restore value
            finetune_model.load_state_dict(E_state)
            C1.load_state_dict(C1_state)
            C2.load_state_dict(C2_state)
            finetune_model.zero_grad()
            pretrain_model.zero_grad()
            C1.zero_grad()
            C2.zero_grad()
            # save model
            save_checkpoint(config, epoch, pretrain_model, finetune_model, C1, C2, 0., pretrain_optimizer,
                            finetune_optimizer, C_optimizer, logger)


def train_one_epoch(config, pretrain_model, E, C1, C2, src_train_loader,
                    tgt_train_loader, pretrain_optim, E_optim, C_optim, pretrain_scheduler, finetune_scheduler,
                    C_scheduler, epoch):
    # 需要更换样本吗
    if on_mac:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    eta = 0.01
    pretrain_model.eval()
    E.train()
    C1.train()
    C2.train()
    finetune_step = min(len(src_train_loader), len(tgt_train_loader))

    train_pred_all = []
    train_all = []
    correct = 0
    total = 0

    time_start_per_epoch = time.time()
    for batch_idx, data in enumerate(zip(src_train_loader, tgt_train_loader)):
        (data_s, mask_s, label_s), (data_t, mask_t, label_t) = data
        assert data_s.shape == data_t.shape, '数据形状不一致'
        data_loader = [[], []]
        for s, t in [[data_s, data_t], [mask_s, mask_t], [label_s, label_t]]:
            s_1, s_2 = s.chunk(2, 0)
            t_1, t_2 = t.chunk(2, 0)
            data_loader[0].append(torch.cat((s_1, t_2)))
            data_loader[1].append(torch.cat((t_1, s_2)))
        if not on_mac:
            data_s, mask_s, label_s = data_s.cuda(), mask_s.cuda(), label_s.cuda()
            data_t, mask_t, label_t = data_t.cuda(), mask_t.cuda(), label_t.cuda()
        data_all = Variable(torch.cat((data_s, data_t), 0))
        # data_all = data_all.type(torch.LongTensor)
        label_s = label_s.long()
        label_s = Variable(label_s)
        bs = len(label_s)
        '''' refactor '''
        pretrain_model.train()
        pretrain_optim.zero_grad()

        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()

        start = time.time()
        end = time.time()
        # index_count = 0
        for idx, (img, mask, _) in enumerate(data_loader):
            # index_count += 1
            # non-blocking 不会堵塞与其无关的的事情
            # img size 128 192 192
            # mask size 128 48 48
            # 遮盖比率为0.75
            if not on_mac:
                img = img.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
            # 从模型的结果得到一个loss
            loss = pretrain_model(img, mask)
            # 更新参数
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(pretrain_model.parameters())
            pretrain_optim.step()
            # lr_scheduler.step_update(epoch * num_steps + idx)
            if not on_mac:
                torch.cuda.synchronize()

            loss_meter.update(loss.item(), img.size(0))
            norm_meter.update(grad_norm)
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                lr = pretrain_optim.param_groups[0]['lr']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
            pretrain_scheduler.step_update(epoch * finetune_step * 2 + batch_idx * 2 + idx)
        epoch_time = time.time() - start
        # logger.info(f"INDEX_COUNT {epoch} index_count is {index_count}")
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

        """source domain discriminative"""
        # Step A train all networks to minimize loss on source
        E_optim.zero_grad()
        C_optim.zero_grad()
        temp_d = copy.deepcopy(pretrain_model.state_dict())
        output = E(data_all, pretrain_model)
        # 输出size是64 512
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
        loss1 = criterion(output_s1, label_s)
        loss2 = criterion(output_s2, label_s)

        all_loss = loss1 + loss2 + 0.01 * entropy_loss
        all_loss.backward()
        E_optim.step()
        C_optim.step()

        """target domain discriminative"""
        # Step B train classifier to maximize discrepancy
        E_optim.zero_grad()
        C_optim.zero_grad()

        output = E(data_all, pretrain_model)
        output1 = C1(output)
        output2 = C2(output)
        output_s1 = output1[:bs, :]
        output_s2 = output2[:bs, :]
        output_t1 = output1[bs:, :]
        output_t2 = output2[bs:, :]
        output_t1 = F.softmax(output_t1, dim=1)
        output_t2 = F.softmax(output_t2, dim=1)

        loss1 = criterion(output_s1, label_s)
        loss2 = criterion(output_s2, label_s)
        entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
        loss_dis = cdd(output_t1, output_t2)

        F_loss = loss1 + loss2 - eta * loss_dis + 0.01 * entropy_loss
        F_loss.backward()
        C_optim.step()

        # Step C train genrator to minimize discrepancy
        NUM_K = 4
        for i in range(NUM_K):
            E.zero_grad()
            C_optim.zero_grad()

            output = E(data_all, pretrain_model)
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
            D_loss = eta * loss_dis + 0.01 * entropy_loss

            D_loss.backward()
            E_optim.step()
        # scheduler step up
        finetune_scheduler.step_update(epoch * finetune_step + batch_idx)
        C_scheduler.step_update(epoch * finetune_step + batch_idx)
    logger.info('Train Ep: {} \tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f} '.format(
        epoch, loss1.item(), loss2.item(), loss_dis.item(), entropy_loss.item()))
    time_end_per_epoch = time.time()
    logger.info(f'time_{epoch}_epoch:{(time_end_per_epoch - time_start_per_epoch)}')


def pretrain_train_one_epoch(config, model, data_loader, optimizer, epoch):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    # index_count = 0
    for idx, (img, mask, _) in enumerate(data_loader):
        # index_count += 1
        # non-blocking 不会堵塞与其无关的的事情
        # img size 128 192 192
        # mask size 128 48 48
        # 遮盖比率为0.75
        if not on_mac:
            img = img.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
        # 从模型的结果得到一个loss
        loss = model(img, mask)
        # 更新参数
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        # lr_scheduler.step_update(epoch * num_steps + idx)
        if not on_mac:
            torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    # logger.info(f"INDEX_COUNT {epoch} index_count is {index_count}")
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def finetune_train_epochs(config, pretrain_model, E, C1, C2, src_train_loader, tgt_train_loader, E_optim, C_optim, finetune_scheduler, C_scheduler,
                          epoch, finetune_epochs):
    # 需要更换样本吗
    if on_mac:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    eta = 0.01
    pretrain_model.eval()
    E.train()
    C1.train()
    C2.train()
    finetune_step = min(len(src_train_loader), len(tgt_train_loader))

    train_pred_all = []
    train_all = []
    correct = 0
    total = 0

    time_start_per_epoch = time.time()
    for ft_epoch in range(finetune_epochs):
        for batch_idx, data in enumerate(zip(src_train_loader, tgt_train_loader)):
            (data_s, mask_s, label_s), (data_t, mask_t, label_t) = data
            if not on_mac:
                data_s, mask_s, label_s = data_s.cuda(), mask_s.cuda(), label_s.cuda()
                data_t, mask_t, label_t = data_t.cuda(), mask_t.cuda(), label_t.cuda()
            data_all = Variable(torch.cat((data_s, data_t), 0))
            # data_all = data_all.type(torch.LongTensor)
            label_s = label_s.long()
            label_s = Variable(label_s)
            bs = len(label_s)

            """source domain discriminative"""
            # Step A train all networks to minimize loss on source
            E_optim.zero_grad()
            C_optim.zero_grad()

            output = E(data_all, pretrain_model)
            # 输出size是64 512
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
            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)

            all_loss = loss1 + loss2 + 0.01 * entropy_loss
            all_loss.backward()
            E_optim.step()
            C_optim.step()

            """target domain discriminative"""
            # Step B train classifier to maximize discrepancy
            E_optim.zero_grad()
            C_optim.zero_grad()

            output = E(data_all, pretrain_model)
            output1 = C1(output)
            output2 = C2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1 = F.softmax(output_t1, dim=1)
            output_t2 = F.softmax(output_t2, dim=1)

            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)
            entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
            loss_dis = cdd(output_t1, output_t2)

            F_loss = loss1 + loss2 - eta * loss_dis + 0.01 * entropy_loss
            F_loss.backward()
            C_optim.step()

            # Step C train genrator to minimize discrepancy
            NUM_K = 4
            for i in range(NUM_K):
                E.zero_grad()
                C_optim.zero_grad()

                output = E(data_all, pretrain_model)
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
                D_loss = eta * loss_dis + 0.01 * entropy_loss

                D_loss.backward()
                E_optim.step()
            # scheduler step up
            finetune_scheduler.step_update(ft_epoch * finetune_step + batch_idx)
            C_scheduler.step_update(ft_epoch * finetune_step + batch_idx)
        logger.info('Train Ep: {} \tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f} '.format(
            epoch, loss1.item(), loss2.item(), loss_dis.item(), entropy_loss.item()))
        E_lr_1 = E_optim.param_groups[0]['lr']
        E_lr_2 = E_optim.param_groups[1]['lr']

        C_lr_1 = C_optim.param_groups[0]['lr']
        C_lr_2 = C_optim.param_groups[1]['lr']
        C_lr_3 = C_optim.param_groups[2]['lr']
        C_lr_4 = C_optim.param_groups[3]['lr']
        logger.info('E_lr_1: {:.6f}\t E_lr_2:{:.6f}'.format(E_lr_1,E_lr_2))
        logger.info('C_lr_1: {:.6f}\t C_lr_2:{:.6f}\t C_lr_3:{:.6f}\t C_lr_4:{:.6f}'.format(C_lr_1, C_lr_2, C_lr_3, C_lr_4))
    time_end_per_epoch = time.time()
    logger.info(f'time_{epoch}_epoch:{(time_end_per_epoch - time_start_per_epoch)}')

def finetune_train_epochs_restore(config, pretrain_model, E, C1, C2, src_train_loader, tgt_train_loader, E_optim, C_optim, epoch, finetune_epochs):
    finetune_optimizer = build_optimizer(config, E, logger, is_pretrain=False)
    C_optimizer = build_optimizer_c(C1, C2, config, logger)
    sche_length = min(len(src_train_loader), len(tgt_train_loader))
    finetune_scheduler = build_finetune_scheduler(config, finetune_optimizer, sche_length, ft_epochs=finetune_epochs)
    C_scheduler = build_finetune_scheduler(config, C_optimizer, sche_length, ft_epochs=finetune_epochs)
    # 复制一下原始双分类器的值
    pretrain_model_state = copy.deepcopy(pretrain_model.state_dict())
    E_state = copy.deepcopy(E.state_dict())

    # 需要更换样本吗
    if on_mac:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    eta = 0.01
    pretrain_model.eval()
    E.train()
    C1.train()
    C2.train()
    finetune_step = min(len(src_train_loader), len(tgt_train_loader))

    train_pred_all = []
    train_all = []
    correct = 0
    total = 0

    time_start_per_epoch = time.time()
    for i in range(finetune_epochs):
        for batch_idx, data in enumerate(zip(src_train_loader, tgt_train_loader)):
            (data_s, mask_s, label_s), (data_t, mask_t, label_t) = data
            if not on_mac:
                data_s, mask_s, label_s = data_s.cuda(), mask_s.cuda(), label_s.cuda()
                data_t, mask_t, label_t = data_t.cuda(), mask_t.cuda(), label_t.cuda()
            data_all = Variable(torch.cat((data_s, data_t), 0))
            # data_all = data_all.type(torch.LongTensor)
            label_s = label_s.long()
            label_s = Variable(label_s)
            bs = len(label_s)

            """source domain discriminative"""
            # Step A train all networks to minimize loss on source
            E_optim.zero_grad()
            C_optim.zero_grad()

            output = E(data_all, pretrain_model)
            # 输出size是64 512
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
            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)

            all_loss = loss1 + loss2 + 0.01 * entropy_loss
            all_loss.backward()
            E_optim.step()
            C_optim.step()

            """target domain discriminative"""
            # Step B train classifier to maximize discrepancy
            E_optim.zero_grad()
            C_optim.zero_grad()

            output = E(data_all, pretrain_model)
            output1 = C1(output)
            output2 = C2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1 = F.softmax(output_t1, dim=1)
            output_t2 = F.softmax(output_t2, dim=1)

            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)
            entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
            loss_dis = cdd(output_t1, output_t2)

            F_loss = loss1 + loss2 - eta * loss_dis + 0.01 * entropy_loss
            F_loss.backward()
            C_optim.step()

            # Step C train genrator to minimize discrepancy
            NUM_K = 4
            for i in range(NUM_K):
                E.zero_grad()
                C_optim.zero_grad()

                output = E(data_all, pretrain_model)
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
                D_loss = eta * loss_dis + 0.01 * entropy_loss

                D_loss.backward()
                E_optim.step()
            # scheduler step up
            finetune_scheduler.step_update(epoch * finetune_step + batch_idx)
            C_scheduler.step_update(epoch * finetune_step + batch_idx)
        logger.info('Train Ep: {} \tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f} '.format(
            epoch, loss1.item(), loss2.item(), loss_dis.item(), entropy_loss.item()))
    time_end_per_epoch = time.time()
    logger.info(f'time_{epoch}_epoch:{(time_end_per_epoch - time_start_per_epoch)}')

    # 恢复之前的参数
    pretrain_model.load_state_dict(pretrain_model_state)
    E.load_state_dict(E_state)

def eval_one_epoch(config, pretrain_model, E, C1, C2, test_loader):
    pretrain_model.eval()
    E.eval()
    C1.eval()
    C2.eval()
    val_pred_all = []
    val_all = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (valX, maskX, valY) in enumerate(test_loader):
            if not on_mac:
                valX, valY = valX.cuda(), valY.cuda()
            output = E(valX, pretrain_model)
            output1 = C1(output)
            output2 = C2(output)
            output_add = output1 + output2
            _, predicted = torch.max(output_add.data, 1)
            total += valY.size(0)
            val_all = np.concatenate([val_all, valY.data.cpu().numpy()])
            val_pred_all = np.concatenate([val_pred_all, predicted.cpu().numpy()])
            correct += predicted.eq(valY.data.view_as(predicted)).cpu().sum().item()
        test_accuracy = 100. * correct / total

        # acc[iDataSet] = test_accuracy
        # # OA = acc
        # C = metrics.confusion_matrix(val_all, val_pred_all)
        # A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
        # K[iDataSet] = metrics.cohen_kappa_score(val_all, val_pred_all)
        logger.info('\tval_Accuracy: {}/{} ({:.2f}%)\t'.format(correct, total, 100. * correct / total))


if __name__ == '__main__':
    _, config = parse_option()
    on_mac = False

    # C:/ProgramData/Anaconda3/envs/CGDM/Lib/site-packages/apex/amp/_amp_state.py 修改了调用问题

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
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    main(config)
