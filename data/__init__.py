import math

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset, TensorDataset, DataLoader

from data.data_finetune import get_finetune_dataloader
from data.data_pretrain import get_pretrain_dataloader
from data.utils import get_mask_dataloader, get_all_data, to_group, get_sample_data, HsiMaskGenerator, \
    HsiMaskTensorDataSet, HsiDataset
from model.Trans_BCDM_A.utils_A import cubeData, cubeData1, get_sample_data_without_train_val, \
    get_sample_data_spatial_spectral
from utils import check_dataset


def get_hsi_spatial_spectral_dataloader(config):
    '''
    必须要保证 空间和光谱的信息 是同一元素上的
    :param config:
    :return:

    '''
    spatial_half_width = config.DATA.SPATIAL.HALF_WIDTH
    spectral_half_width = config.DATA.SPECTRAL.HALF_WIDTH
    dataset = check_dataset(config)
    if dataset == 'indian' or dataset == 'shanghai-hangzhou':
        img_source, label_source, img_target, label_target = cubeData(config.DATA.DATA_SOURCE_PATH,
                                                                      config.DATA.LABEL_SOURCE_PATH,
                                                                      config.DATA.DATA_TARGET_PATH,
                                                                      config.DATA.LABEL_TARGET_PATH, dataset)
    else:
        img_source, label_source = cubeData1(config.DATA.DATA_SOURCE_PATH, config.DATA.LABEL_SOURCE_PATH, dataset)
        img_target, label_target = cubeData1(config.DATA.DATA_TARGET_PATH, config.DATA.LABEL_TARGET_PATH, dataset)
    source_spatial_samples, source_spectral_samples, source_labels = get_sample_data_spatial_spectral(img_source,
                                                                                                      label_source,
                                                                                                      spatial_half_width,
                                                                                                      spectral_half_width,
                                                                                                      0)
    target_spatial_samples, target_spectral_samples, target_labels = get_sample_data_spatial_spectral(img_target,
                                                                                                      label_target,
                                                                                                      spatial_half_width,
                                                                                                      spectral_half_width,
                                                                                                      0)
    # 空间数据进行tranformer和空间pad
    spatial_patches = math.ceil((spatial_half_width * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE) ** 2
    transform_spatial = HsiMaskGenerator(config.DATA.MASK_RATIO, spatial_patches,
                                 mask_patch_size=1)
    transform_spectral = HsiMaskGenerator(config.DATA.MASK_RATIO, target_spectral_samples.shape[1],
                                 mask_patch_size=config.DATA.MASK_PATCH_SIZE)
    B, C, H, W = source_spatial_samples.shape
    pad_pixel = math.ceil((spatial_half_width * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE) * config.DATA.SPATIAL.PATCH_SIZE - (
            spatial_half_width * 2 + 1)
    column_split = row_spilt = math.ceil((spatial_half_width * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE)
    # to B,C,36,36
    source_samples = np.pad(source_spatial_samples, ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
                                             (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    # test_img = np.pad(test_img, ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
    #                              (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    target_samples = np.pad(target_spatial_samples, ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
                                             (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    # to B 36 36 C
    source_samples, source_labels, target_samples, target_labels = torch.tensor(source_samples), torch.tensor(
        source_labels), torch.tensor(target_samples), torch.tensor(target_labels)
    rearrange(source_samples, 'b c (h1 h) (w1 w) -> b (h w) (c h1 w1)', h1=column_split, h2=row_spilt)
    HsiDataset(source_spatial_samples,source_spectral_samples,source_labels,transform_spatial, transform_spectral)
    # rearrange(test_img, 'b c (h1 h) (w1 w) -> b (h w) (c h1 w1)', h1=column_split, h2=row_spilt)
    rearrange(target_samples, 'b c (h1 h) (w1 w) -> b (h w) (c h1 w1)', h1=column_split, h2=row_spilt)

    # 定义一个新的dataset 包含空间信息，光谱信息，标签，mask空间，mask光谱
    HsiMaskTensorDataSet

    source_samples = torch.range(1, 128 * 48 * 33 * 33).reshape((128, 48, 33, 33)).numpy()
    source_labels = torch.full((128,), 1.0).numpy()
    target_samples = torch.range(1, 128 * 48 * 33 * 33).reshape((128, 48, 33, 33)).numpy()
    target_labels = torch.full((128,), 1.0).numpy()
    test_img = torch.range(1, 128 * 48 * 33 * 33).reshape((128, 48, 33, 33)).numpy()
    test_label = torch.full((128,), 1.0).numpy()
    return None


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
    source_samples, source_labels = get_sample_data_without_train_val(img_source, label_source, halfwidth, config.DATA.SAMPLE_NUM)
    target_samples, target_labels = get_sample_data_without_train_val(img_target, label_target, halfwidth, 0)

    # 微调所使用的数据
    # test_img, test_label = get_all_data(img_target, label_target, halfwidth)  # 目标域全部样本
    pathes = math.ceil((halfwidth*2+1) / config.DATA.SPATIAL.PATCH_SIZE)**2
    transform = HsiMaskGenerator(config.DATA.MASK_RATIO, pathes,
                                 mask_patch_size=1)
    # source_samples = torch.range(1, 128 * 48 * 33 * 33).reshape((128, 48, 33, 33)).numpy()
    # source_labels = torch.full((128,), 1.0).numpy()
    # target_samples = torch.range(1, 128 * 48 * 33 * 33).reshape((128, 48, 33, 33)).numpy()
    # target_labels = torch.full((128,), 1.0).numpy()
    # test_img = torch.range(1, 128 * 48 * 33 * 33).reshape((128, 48, 33, 33)).numpy()
    # test_label = torch.full((128,), 1.0).numpy()
    B, C, H, W = source_samples.shape
    pad_pixel = math.ceil((halfwidth * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE) * config.DATA.SPATIAL.PATCH_SIZE - (
                halfwidth * 2 + 1)
    column_split = row_spilt = math.ceil((halfwidth * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE)
    # to B,C,36,36
    source_samples = np.pad(source_samples, ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
                                             (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    # test_img = np.pad(test_img, ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
    #                              (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    target_samples = np.pad(target_samples, ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
                                             (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    # to B 36 36 C
    source_samples, source_labels, target_samples, target_labels = torch.tensor(source_samples), torch.tensor(source_labels), torch.tensor(target_samples), torch.tensor(target_labels)
    rearrange(source_samples, 'b c (h1 h) (w1 w) -> b (h w) (c h1 w1)', h1=column_split, h2=row_spilt)
    # rearrange(test_img, 'b c (h1 h) (w1 w) -> b (h w) (c h1 w1)', h1=column_split, h2=row_spilt)
    rearrange(target_samples, 'b c (h1 h) (w1 w) -> b (h w) (c h1 w1)', h1=column_split, h2=row_spilt)

    # source_samples = rearrange(source_samples, 'b c (h,h1) (w,w1) -> b (h,w) (c,h1,w1)')
    # test_img = to_group(test_img, config)
    test_dataset = HsiMaskTensorDataSet(target_samples,target_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=4)
    # src_img, src_label = get_sample_data(img_source, label_source, halfwidth, config.DATA.SAMPLE_NUM)
    # src_img = to_group(src_img, config)
    train_dataset = HsiMaskTensorDataSet(source_samples, source_labels, transform=transform)
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
