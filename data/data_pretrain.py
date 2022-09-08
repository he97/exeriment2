import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.utils import HsiMaskGenerator, to_group, HsiMaskTensorDataSet, get_all_data, get_sample_data
from model.Trans_BCDM_A.utils_A import cubeData, cubeData1, get_sample_data_without_train_val
from utils import check_dataset


def get_pretrain_dataloader(config):
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
    test_img = to_group(test_img,config)
    test_dataset = TensorDataset(torch.tensor(test_img), torch.tensor(test_label))
    test_loader = DataLoader(test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=4)
    src_img, src_label = get_sample_data(img_source, label_source, halfwidth, config.DATA.SAMPLE_NUM)
    src_img = to_group(src_img,config)
    train_dataset = TensorDataset(torch.tensor(src_img), torch.tensor(src_label))
    src_train_loader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4)
    tgt_train_loader = DataLoader(test_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=4)

    if target_samples.shape[0] > source_samples.shape[0]:
        ratio = target_samples.shape[0] // source_samples.shape[0]
        source_samples = source_samples.repeat(ratio, axis=0)
        source_labels = source_labels.repeat(ratio, axis=0)
    else:
        ratio = source_samples.shape[0] // target_samples.shape[0]
        target_samples = target_samples.repeat(ratio, axis=0)
        target_labels = target_labels.repeat(ratio, axis=0)
    l_sample = []
    l_label = []
    for i in range(min(source_samples.shape[0], target_samples.shape[0])):
        l_sample.append(source_samples[i])
        l_sample.append(target_samples[i])
        l_label.append(source_labels[i])
        l_label.append(target_labels[i])
    all_samples = np.array(l_sample)
    all_labels = np.array(l_label)

    transform = HsiMaskGenerator(config.DATA.MASK_RATIO, all_samples.shape[1],
                                 mask_patch_size=config.DATA.MASK_PATCH_SIZE)
    all_samples = to_group(all_samples, config)
    dataset = HsiMaskTensorDataSet(torch.tensor(all_samples), torch.tensor(all_labels), transform=transform)
    data_loader = DataLoader(dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=0, sampler=None,
                             pin_memory=True, drop_last=True)

    return data_loader,test_loader,src_train_loader,tgt_train_loader
