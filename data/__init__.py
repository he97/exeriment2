import numpy as np
from torch.utils.data import Dataset

from data.data_finetune import get_finetune_dataloader
from data.data_pretrain import get_pretrain_dataloader
from data.utils import get_mask_dataloader


def get_dataloader(config,is_pretrain:bool = True):
    if is_pretrain:
        return get_pretrain_dataloader(config)
    else:
        return get_finetune_dataloader(config)

def get_virtual_dataloader(config,size):
    return get_mask_dataloader(config,size)