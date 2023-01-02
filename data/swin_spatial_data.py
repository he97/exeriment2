import math

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader

from data import get_all_data, get_pca_data, HsiMaskTensorDataSet
from model.Trans_BCDM_A.utils_A import cubeData, cubeData1, get_sample_data_without_train_val
from utils import check_dataset


def get_hsi_spatial_swin_dataloader(config):
    from data import HsiMaskGenerator, HsiMaskTensorDataSet
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
    source_samples, source_labels = get_sample_data_without_train_val(img_source, label_source, halfwidth,
                                                                      config.DATA.SAMPLE_NUM)
    target_samples, target_labels = get_sample_data_without_train_val(img_target, label_target, halfwidth, 0)

    # 微调所使用的数据
    # test_img, test_label = get_all_data(img_target, label_target, halfwidth)  # 目标域全部样本
    pathes = math.ceil((halfwidth * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE) ** 2
    transform = HsiMaskGenerator(config.DATA.SPATIAL_MASK_RATIO, pathes,
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
    source_samples, source_labels, target_samples, target_labels = torch.tensor(source_samples), torch.tensor(
        source_labels), torch.tensor(target_samples), torch.tensor(target_labels)
    # source_samples = rearrange(source_samples, 'b c (h1 h) (w1 w) -> b (h1 w1) (c h w)', h1=column_split, w1=row_spilt)
    # # rearrange(test_img, 'b c (h1 h) (w1 w) -> b (h w) (c h1 w1)', h1=column_split, h2=row_spilt)
    # target_samples = rearrange(target_samples, 'b c (h1 h) (w1 w) -> b (h1 w1) (c h w)', h1=column_split, w1=row_spilt)

    # source_samples = rearrange(source_samples, 'b c (h,h1) (w,w1) -> b (h,w) (c,h1,w1)')
    # test_img = to_group(test_img, config)
    test_dataset = HsiMaskTensorDataSet(target_samples, target_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=4)
    # src_img, src_label = get_sample_data(img_source, label_source, halfwidth, config.DATA.SAMPLE_NUM)
    # src_img = to_group(src_img, config)
    train_dataset = HsiMaskTensorDataSet(source_samples, source_labels, transform=transform)
    src_train_loader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4,
                                  drop_last=True)
    tgt_train_loader = DataLoader(test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4,
                                  drop_last=True)

    return test_loader, src_train_loader, tgt_train_loader


def get_hsi_spatial_pca_swin_dataloader(config):
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
    source_samples, source_labels = get_sample_data_without_train_val(img_source, label_source, halfwidth,
                                                                      config.DATA.SAMPLE_NUM)
    target_samples, target_labels = get_sample_data_without_train_val(img_target, label_target, halfwidth, 0)
    length = config.DATA.SAMPLE_NUM*config.DATA.CLASS_NUM
    pca_data = np.concatenate((source_samples,target_samples[:length]))
    pca_label = np.concatenate((source_labels,target_labels[:length]))
    pca_data = get_pca_data(pca_data, config.DATA.SPATIAL.COMPONENT_NUM)
    all_target_pca_data = get_pca_data(target_samples, config.DATA.SPATIAL.COMPONENT_NUM)
    source_pca_samples = pca_data[:length]
    target_pca_samples = pca_data[length:]
    source_pca_label = pca_label[:length]
    target_pca_label = pca_label[length:]
    # 微调所使用的数据
    # test_img, test_label = get_all_data(img_target, label_target, halfwidth)  # 目标域全部样本
    patches = math.ceil((halfwidth * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE) ** 2
    transform = HsiMaskGenerator(config.DATA.MASK_RATIO, patches,
                                 mask_patch_size=1)
    B, C, H, W = source_samples.shape
    pad_pixel = math.ceil((halfwidth * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE) * config.DATA.SPATIAL.PATCH_SIZE - (
            halfwidth * 2 + 1)
    column_split = row_spilt = math.ceil((halfwidth * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE)
    # to B,C,36,36
    source_pca_samples = np.pad(source_pca_samples, ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
                                             (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    # test_img = np.pad(test_img, ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
    #                              (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    target_pca_samples = np.pad(target_pca_samples, ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
                                             (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    all_target_pca_data = np.pad(all_target_pca_data,
                                ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
                                 (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    # to B 81 C
    # source_samples, source_labels, target_samples, target_labels = torch.tensor(source_samples), torch.tensor(
    #     source_labels), torch.tensor(target_samples), torch.tensor(target_labels)
    all_target_pca_data = torch.tensor(all_target_pca_data)
    source_pca_samples,source_pca_label = torch.tensor(source_pca_samples),torch.tensor(source_pca_label)
    target_pca_samples, target_pca_label = torch.tensor(target_pca_samples), torch.tensor(target_pca_label)
    # source_pca_samples = rearrange(source_pca_samples, 'b c (h1 h) (w1 w) -> b (h1 w1) (c h w)', h1=column_split, w1=row_spilt)
    # rearrange(test_img, 'b c (h1 h) (w1 w) -> b (h w) (c h1 w1)', h1=column_split, h2=row_spilt)
    # target_pca_samples = rearrange(target_pca_samples, 'b c (h1 h) (w1 w) -> b (h1 w1) (c h w)', h1=column_split, w1=row_spilt)
    # all_target_pca_data = rearrange(all_target_pca_data, 'b c (h1 h) (w1 w) -> b (h1 w1) (c h w)', h1=column_split, w1=row_spilt)

    # source_samples = rearrange(source_samples, 'b c (h,h1) (w,w1) -> b (h,w) (c,h1,w1)')
    # test_img = to_group(test_img, config)
    test_dataset = HsiMaskTensorDataSet(all_target_pca_data, target_labels, transform=transform)
    target_pca_dataset = HsiMaskTensorDataSet(target_pca_samples, target_pca_label, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=4)
    # src_img, src_label = get_sample_data(img_source, label_source, halfwidth, config.DATA.SAMPLE_NUM)
    # src_img = to_group(src_img, config)
    train_pca_dataset = HsiMaskTensorDataSet(source_pca_samples, source_labels, transform=transform)
    src_train_loader = DataLoader(train_pca_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4,
                                  drop_last=True)
    tgt_train_loader = DataLoader(target_pca_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4,
                                  drop_last=True)

    return test_loader, src_train_loader, tgt_train_loader
