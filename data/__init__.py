import math

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset, TensorDataset, DataLoader

from data.data_finetune import get_finetune_dataloader
from data.data_pretrain import get_pretrain_dataloader
from data.utils import get_mask_dataloader, get_all_data, to_group, get_sample_data, HsiMaskGenerator, \
    HsiMaskTensorDataSet
from model.Trans_BCDM_A.utils_A import cubeData, cubeData1, get_sample_data_without_train_val
from utils import check_dataset


def get_hsi_spatial_dataloader(config):
    halfwidth = config.DATA.SPATIAL.HALF_WIDTH
    dataset = check_dataset(config)
    if dataset == 'indian' or dataset == 'shanghai-hangzhou':
        img_source, label_source, img_target, label_target = cubeData(config.DATA.DATA_SOURCE_PATH,
                                                                      config.DATA.LABEL_SOURCE_PATH,
                                                                      config.DATA.DATA_TARGET_PATH,
                                                                      config.DATA.LABEL_TARGET_PATH, dataset)
    else:
        img_source, label_source = cubeData1(config.DATA.DATA_SOURCE_PATH, config.DATA.LABEL_SOURCE_PATH, dataset)
        img_target, label_target = cubeData1(config.DATA.DATA_TARGET_PATH, config.DATA.LABEL_TARGET_PATH, dataset)
    source_samples, source_labels = get_sample_data_without_train_val(img_source, label_source, halfwidth, 0)
    target_samples, target_labels = get_sample_data_without_train_val(img_target, label_target, halfwidth, 0)

    # 微调所使用的数据
    test_img, test_label = get_all_data(img_target, label_target, halfwidth)  # 目标域全部样本
    pathes = math.ceil((halfwidth*2+1) / config.DATA.SPATIAL.PATCH_SIZE)**2
    transform = HsiMaskGenerator(config.DATA.MASK_RATIO, pathes,
                                 mask_patch_size=1)
    B,C,H,W = source_samples.shape
    pad_pixel = math.ceil((halfwidth*2+1) / config.DATA.SPATIAL.PATCH_SIZE)*config.DATA.SPATIAL.PATCH_SIZE - (halfwidth*2+1)
    column_split = row_spilt = math.ceil((halfwidth*2+1) / config.DATA.SPATIAL.PATCH_SIZE)
    # to B,C,36,36
    source_samples = np.pad(source_samples,((0,0),(0,0),(math.floor(pad_pixel/2),math.ceil(pad_pixel/2)),
                                            (math.floor(pad_pixel/2),math.ceil(pad_pixel/2))),'constant')
    test_img = np.pad(test_img, ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
                                             (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    # to B 36 36 C
    source_samples,source_labels,test_img,test_label = torch.tensor(source_samples),torch.tensor(source_labels),torch.tensor(test_img),torch.tensor(test_label)
    rearrange(source_samples, 'b c (h1 h) (w1 w) -> b (h w) (c h1 w1)', h1=column_split, h2=row_spilt)
    source_samples = rearrange(source_samples,'b c (h,h1) (w,w1) -> b (h,w) (c,h1,w1)')
    # test_img = to_group(test_img, config)
    test_dataset = HsiMaskTensorDataSet(torch.tensor(test_img), torch.tensor(test_label), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=4)
    src_img, src_label = get_sample_data(img_source, label_source, halfwidth, config.DATA.SAMPLE_NUM)
    src_img = to_group(src_img, config)
    train_dataset = HsiMaskTensorDataSet(torch.tensor(src_img), torch.tensor(src_label), transform=transform)
    src_train_loader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4,
                                  drop_last=True)
    tgt_train_loader = DataLoader(test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4,
                                  drop_last=True)

    return test_loader, src_train_loader, tgt_train_loader




def get_hsi_spectral_dataloader(config, is_pretrain: bool = True):
    halfwidth = 2
    dataset = check_dataset(config)
    if dataset == 'indian' or dataset == 'shanghai-hangzhou':
        img_source, label_source, img_target, label_target = cubeData(config.DATA.DATA_SOURCE_PATH,
                                                                      config.DATA.LABEL_SOURCE_PATH,
                                                                      config.DATA.DATA_TARGET_PATH,
                                                                      config.DATA.LABEL_TARGET_PATH, dataset)
    else:
        img_source, label_source = cubeData1(config.DATA.DATA_SOURCE_PATH, config.DATA.LABEL_SOURCE_PATH, dataset)
        img_target, label_target = cubeData1(config.DATA.DATA_TARGET_PATH, config.DATA.LABEL_TARGET_PATH, dataset)
    source_samples, source_labels = get_sample_data_without_train_val(img_source, label_source, halfwidth, 0)
    target_samples, target_labels = get_sample_data_without_train_val(img_target, label_target, halfwidth, 0)

    # 微调所使用的数据
    test_img, test_label = get_all_data(img_target, label_target, halfwidth)  # 目标域全部样本
    transform = HsiMaskGenerator(config.DATA.MASK_RATIO, test_img.shape[1],
                                 mask_patch_size=config.DATA.MASK_PATCH_SIZE)
    test_img = to_group(test_img, config)
    test_dataset = HsiMaskTensorDataSet(torch.tensor(test_img), torch.tensor(test_label), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=4)
    src_img, src_label = get_sample_data(img_source, label_source, halfwidth, config.DATA.SAMPLE_NUM)
    src_img = to_group(src_img, config)
    train_dataset = HsiMaskTensorDataSet(torch.tensor(src_img), torch.tensor(src_label), transform=transform)
    src_train_loader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)
    tgt_train_loader = DataLoader(test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)

    return test_loader, src_train_loader, tgt_train_loader
def get_virtual_dataloader(config, size):
    return get_mask_dataloader(config, size)
