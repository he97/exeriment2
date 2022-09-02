import os

import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing


def check_dataset(config):
    if 'houston' in config.DATA.DATA_SOURCE_PATH:
        dataset = 'houston'
    elif 'pavia' in config.DATA.DATA_SOURCE_PATH:
        dataset = 'pavia'
    elif 'indian' in config.DATA.DATA_SOURCE_PATH:
        dataset = 'indian'
    elif 'shanghai-hangzhou' in config.DATA.DATA_SOURCE_PATH:
        dataset = 'shanghai-hangzhou'
    else:
        raise Exception('dataset not support')
    return dataset

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def save_checkpoint(config, epoch, pretrain_model,finrtune_model,C1,C2, max_accuracy, pretrain_optimizer, finetune_optimizer, C_optimizer, logger):
    save_state = {'pretrain_model': pretrain_model.state_dict(),
                  'finetune_model': finrtune_model.state_dict(),
                  'C1': C1.state_dict(),
                  'C2': C2.state_dict(),
                  'pretrain_optimizer': pretrain_optimizer.state_dict(),
                  'finetune_optimizer': finetune_optimizer.state_dict(),
                  'C_optimizer': C_optimizer.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")