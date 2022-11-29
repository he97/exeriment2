import math

import numpy as np
import torch
from einops import rearrange
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, TensorDataset, DataLoader

from data.data_finetune import get_finetune_dataloader
from data.data_pretrain import get_pretrain_dataloader
from data.utils import get_mask_dataloader, get_all_data, to_group, get_sample_data, HsiMaskGenerator, \
    HsiMaskTensorDataSet, HsiDataset, get_pca_data
from model.Trans_BCDM_A.utils_A import cubeData, cubeData1, get_sample_data_without_train_val, \
    get_sample_data_spatial_spectral
from utils import check_dataset
def get_hsi_pca_spatial_spectral_dataloader(config):
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
                                                                                                      config.DATA.SAMPLE_NUM)
    target_spatial_samples, target_spectral_samples, target_labels = get_sample_data_spatial_spectral(img_target,
                                                                                                      label_target,
                                                                                                      spatial_half_width,
                                                                                                      spectral_half_width,
                                                                                                      0)
    length = config.DATA.SAMPLE_NUM * config.DATA.CLASS_NUM
    spatial_pca_data = np.concatenate((source_spatial_samples, target_spatial_samples[:length]))
    # pca_label = np.concatenate((source_labels, target_labels[:length]))
    spatial_pca_data = get_pca_data(spatial_pca_data, config.DATA.SPATIAL.COMPONENT_NUM)
    all_target_spatial_pca_data = get_pca_data(target_spatial_samples, config.DATA.SPATIAL.COMPONENT_NUM)
    source_spatial_pca_samples = spatial_pca_data[:length]
    target_spatial_pca_samples = spatial_pca_data[length:]
    source_spatial_pca_label = pca_label[:length]
    target_spatial_pca_label = pca_label[length:]
    # spatial空间数据进行tranformer和空间pad
    spatial_patches = math.ceil((spatial_half_width * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE) ** 2
    transform_spatial = HsiMaskGenerator(config.DATA.MASK_RATIO, spatial_patches,
                                         mask_patch_size=1)
    transform_spectral = HsiMaskGenerator(config.DATA.MASK_RATIO, target_spectral_samples.shape[1],
                                          mask_patch_size=config.DATA.MASK_PATCH_SIZE)
    # spectral info to group
    source_spectral_samples = to_group(source_spectral_samples, config)
    target_spectral_samples = to_group(target_spectral_samples, config)
    B, C, H, W = source_spatial_samples.shape
    pad_pixel = math.ceil(
        (spatial_half_width * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE) * config.DATA.SPATIAL.PATCH_SIZE - (
                        spatial_half_width * 2 + 1)
    column_split = row_spilt = math.ceil((spatial_half_width * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE)
    # to B,C,36,36
    source_spatial_pca_samples = np.pad(source_spatial_pca_samples,
                                ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
                                 (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    # test_img = np.pad(test_img, ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
    #                              (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    target_spatial_pca_samples = np.pad(target_spatial_pca_samples,
                                ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
                                 (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    all_target_spatial_pca_data = np.pad(all_target_spatial_pca_data,
                                 ((0, 0), (0, 0), (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2)),
                                  (math.floor(pad_pixel / 2), math.ceil(pad_pixel / 2))), 'constant')
    # to B 36 36 C
    source_spatial_samples, source_spectral_samples = torch.tensor(source_spatial_samples), torch.tensor(source_spectral_samples)
    target_spatial_samples, target_spectral_samples = torch.tensor(target_spatial_samples), torch.tensor(target_spectral_samples)
    source_labels, target_labels = torch.tensor(source_labels), torch.tensor(target_labels)
    source_pca_samples, source_pca_label = torch.tensor(source_spatial_pca_samples), torch.tensor(source_spatial_pca_label)
    target_pca_samples, target_pca_label = torch.tensor(target_spatial_pca_samples), torch.tensor(target_spatial_pca_label)
    source_pca_samples = rearrange(source_pca_samples, 'b c (h1 h) (w1 w) -> b (h1 w1) (c h w)', h1=column_split,
                                   w1=row_spilt)
    # rearrange(test_img, 'b c (h1 h) (w1 w) -> b (h w) (c h1 w1)', h1=column_split, h2=row_spilt)
    target_pca_samples = rearrange(target_pca_samples, 'b c (h1 h) (w1 w) -> b (h1 w1) (c h w)', h1=column_split,
                                   w1=row_spilt)
    all_target_spatial_pca_data = rearrange(all_target_spatial_pca_data, 'b c (h1 h) (w1 w) -> b (h1 w1) (c h w)', h1=column_split,
                                    w1=row_spilt)

    # 定义一个新的dataset 包含空间信息，光谱信息，标签，mask空间，mask光谱
    source_dataset = HsiDataset(source_spatial_samples, source_spectral_samples, source_labels, transform_spatial, transform_spectral)
    # rearrange(test_img, 'b c (h1 h) (w1 w) -> b (h w) (c h1 w1)', h1=column_split, h2=row_spilt)
    target_dataset = HsiDataset(target_spatial_samples,target_spectral_samples,target_labels,transform_spatial, transform_spectral)
    test_dataset = HsiDataset(target_spatial_pca_samples,target_spectral_samples,target_labels,transform_spatial,transform_spectral)


    # dataloader
    test_loader = DataLoader(test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=4)
    src_train_loader = DataLoader(source_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4,
                                  drop_last=True)
    tgt_train_loader = DataLoader(target_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4,
                                  drop_last=True)
    return test_loader, src_train_loader, tgt_train_loader
    # source_samples = torch.range(1, 128 * 48 * 33 * 33).reshape((128, 48, 33, 33)).numpy()
    # source_labels = torch.full((128,), 1.0).numpy()
    # target_samples = torch.range(1, 128 * 48 * 33 * 33).reshape((128, 48, 33, 33)).numpy()
    # target_labels = torch.full((128,), 1.0).numpy()
    # test_img = torch.range(1, 128 * 48 * 33 * 33).reshape((128, 48, 33, 33)).numpy()
    # test_label = torch.full((128,), 1.0).numpy()
    # return None
