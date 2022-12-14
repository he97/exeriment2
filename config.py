# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# classifier settings
_C.CLASSIFIER = CN()
_C.CLASSIFIER.MODE = 'base'
_C.CLASSIFIER.ATTENTION = CN()
_C.CLASSIFIER.ATTENTION.DEPTH = 1
_C.CLASSIFIER.ATTENTION.DIM = 1024
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.SPATIAL = CN()
_C.DATA.SPATIAL.HALF_WIDTH = 16
_C.DATA.SPATIAL.PATCH_SIZE = 4
_C.DATA.SPATIAL.COMPONENT_NUM = 3
_C.DATA.SPATIAL.PCA = False
_C.DATA.SPECTRAL = CN()
_C.DATA.SPECTRAL.HALF_WIDTH = 2
_C.DATA.SPECTRAL.CHANNEL_DIM = 48
# _C.DATA.SPATIAL.PATCH_SIZE = 1
# sum of epoch
_C.DATA.EPOCHS = 80
    # epochs = 80
#     learning rate
_C.DATA.LEARNING_RATE = 1e-4
    # lr = 1e-4
# gamma
_C.DATA.GAMMA = 0.9
    # gamma = 0.9
# seed
_C.DATA.SEED = 0
    # seed = 0
# halfwidth
#*****
#*****
#**0**
#*****
#*****
_C.DATA.HALFWIDTH = 2
    # HalfWidth = 2
# 这个的用处 只有一个每种类找多少个样本 没用 先不写
_C.DATA.SAMPLE_NUM = 180
    # SAMPLE_NUM = 180
    # 种类数
_C.DATA.N_CLASS = 32
    # nClass = 16
    # 波段数
_C.DATA.CHANNEL_DIM = 48
# 这个命名是不对的，表示的每个channel表示为长度为多少的特征向量
_C.DATA.PATCH_DIM = 512
    # dim = 112
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# Path to dataset, could be overwritten by command line argument
_C.DATA.PATCH_SIZE = 5
_C.DATA.DATA_PATH = ''
_C.DATA.MODE = 'spatial+spectral'
_C.DATA.DATA_SOURCE_PATH = ''
_C.DATA.DATA_TARGET_PATH = ''
_C.DATA.LABEL_TARGET_PATH = ''
_C.DATA.LABEL_SOURCE_PATH = ''
_C.DATA.IS_MASK = True
# is hsi
_C.IS_HSI = True
# id distrubition
_C.IS_DIST = False
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6
_C.DATA.SPATIAL_MASK_RATIO = 0.6

_C.DATA.CLASS_NUM = 7
_C.DATA.DIM = 48
_C.DATA.SPATIAL_ORIGIN = False
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
# spatial and spectral model settings
_C.SPATIAL = CN()
_C.SPATIAL.TYPE = 'swin'
_C.SPECTRAL = CN()
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
# llearning rate
_C.MODEL.LEARNING_RATE = 1e-4
# patch_dim
_C.MODEL.SPATIAL_PATCH_DIM = 1024
_C.MODEL.SPECTRAL_PATCH_DIM = 1024
_C.MODEL.CLASSIFIER_IN_DIM = 1024


# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
# peng swin end_stage -> output dim
_C.MODEL.SWIN.STAGE_DIM = [96, 192, 384, 768]
_C.MODEL.SWIN.END_STAGE = 3


# Vision Transformer parameters
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCH_SIZE = 16
_C.MODEL.VIT.IN_CHANS = 3
_C.MODEL.VIT.EMBED_DIM = 768
_C.MODEL.VIT.DEPTH = 12
_C.MODEL.VIT.NUM_HEADS = 12
_C.MODEL.VIT.MLP_RATIO = 4
_C.MODEL.VIT.QKV_BIAS = True
_C.MODEL.VIT.INIT_VALUES = 0.1
_C.MODEL.VIT.USE_APE = False
_C.MODEL.VIT.USE_RPB = False
_C.MODEL.VIT.USE_SHARED_RPB = True
_C.MODEL.VIT.USE_MEAN_POOLING = False

# Dtransformer Transformer parameters
_C.MODEL.Dtransformer = CN()
_C.MODEL.Dtransformer.PATCH_SIZE = 5
_C.MODEL.Dtransformer.IN_CHANS = 48
_C.MODEL.Dtransformer.EMBED_DIM = 96
_C.MODEL.Dtransformer.DEPTHS = [2, 2, 6, 2]
_C.MODEL.Dtransformer.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.Dtransformer.WINDOW_SIZE = 7
_C.MODEL.Dtransformer.MLP_RATIO = 4.
_C.MODEL.Dtransformer.QKV_BIAS = True
_C.MODEL.Dtransformer.QK_SCALE = None
_C.MODEL.Dtransformer.APE = False
_C.MODEL.Dtransformer.PATCH_NORM = True
_C.MODEL.Dtransformer.DEPTH = 3
_C.MODEL.Dtransformer.SPATIAL_DEPTH = 3
_C.MODEL.Dtransformer.PATCH_DIM = 512

# SPECTRAL_FORMER Transformer parameters
_C.MODEL.SPECTRAL_FORMER = CN()
_C.MODEL.SPECTRAL_FORMER.PATCH_SIZE = 5
_C.MODEL.SPECTRAL_FORMER.IN_CHANS = 48
_C.MODEL.SPECTRAL_FORMER.EMBED_DIM = 96
_C.MODEL.SPECTRAL_FORMER.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SPECTRAL_FORMER.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SPECTRAL_FORMER.WINDOW_SIZE = 7
_C.MODEL.SPECTRAL_FORMER.MLP_RATIO = 4.
_C.MODEL.SPECTRAL_FORMER.QKV_BIAS = True
_C.MODEL.SPECTRAL_FORMER.QK_SCALE = None
_C.MODEL.SPECTRAL_FORMER.APE = False
_C.MODEL.SPECTRAL_FORMER.PATCH_NORM = True
_C.MODEL.SPECTRAL_FORMER.DEPTH = 3
_C.MODEL.SPECTRAL_FORMER.SPATIAL_DEPTH = 3
_C.MODEL.SPECTRAL_FORMER.PATCH_DIM = 512
_C.MODEL.SPECTRAL_FORMER.MODE = "ViT"


# SSFTTNET Transformer parameters
_C.MODEL.SSFTTNET = CN()
_C.MODEL.SSFTTNET.PATCH_SIZE = 5
_C.MODEL.SSFTTNET.IN_CHANS = 48
_C.MODEL.SSFTTNET.EMBED_DIM = 96
_C.MODEL.SSFTTNET.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SSFTTNET.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SSFTTNET.WINDOW_SIZE = 7
_C.MODEL.SSFTTNET.MLP_RATIO = 4.
_C.MODEL.SSFTTNET.QKV_BIAS = True
_C.MODEL.SSFTTNET.QK_SCALE = None
_C.MODEL.SSFTTNET.APE = False
_C.MODEL.SSFTTNET.PATCH_NORM = True
_C.MODEL.SSFTTNET.DEPTH = 3
_C.MODEL.SSFTTNET.PATCH_DIM = 512


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------



_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 10
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 8e-4
_C.TRAIN.WARMUP_LR = 5e-6
_C.TRAIN.MIN_LR = 5e-7
_C.TRAIN.FINETUNE = CN()
_C.TRAIN.FINETUNE.WARMUP_EPOCHS = 10
_C.TRAIN.FINETUNE.WEIGHT_DECAY = 0.05
_C.TRAIN.FINETUNE.BASE_LR = 5e-6
_C.TRAIN.FINETUNE.WARMUP_LR = 5e-7
_C.TRAIN.FINETUNE.MIN_LR = 5e-8
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# punish item * eta
_C.TRAIN.ETA = 0.01
# refactor loss item * eta
_C.TRAIN.RF_ETA = 1
_C.TRAIN.SPATIAL_RF_ETA = 1
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

# [SimMIM] path to pre-trained model
_C.PRETRAINED = ''


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False
    def _check_args_without_eval(name):
        if hasattr(args, name):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('pretrained'):
        config.PRETRAINED = args.pretrained
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True
    if _check_args('data_source_path'):
        config.DATA.DATA_SOURCE_PATH = args.data_source_path
    if _check_args('data_target_path'):
        config.DATA.DATA_TARGET_PATH = args.data_target_path
    if _check_args('label_source_path'):
        config.DATA.LABEL_SOURCE_PATH = args.label_source_path
    if _check_args('label_target_path'):
        config.DATA.LABEL_TARGET_PATH = args.label_target_path
    if _check_args('is_dist'):
        config.IS_DIST = args.is_dist
    if _check_args('is_hsi'):
        config.IS_HSI = args.is_HSI
    if _check_args('eta'):
        config.TRAIN.ETA = args.eta
    if _check_args_without_eval('mask_ratio'):
        config.DATA.MASK_RATIO = args.mask_ratio
    if _check_args_without_eval("refactor_eta"):
        config.TRAIN.RF_ETA = args.refactor_eta
    if _check_args("attention_depth"):
        type = config.MODEL.TYPE
        if type == "SPECTRAL_FORMER":
            # config.MODEL.eval(config.MODEL.TYPE).DEPTH = args.attention_depth
            config.MODEL.SPECTRAL_FORMER.DEPTH = args.attention_depth
        elif type == "Dtransformer":
            config.MODEL.Dtransformer.DEPTH = args.attention_depth
        elif type == 'swin':
            pass
        else:
            raise Exception(f"not support type:{type}")
        # config.MODEL.eval(config.MODEL.TYPE).DEPTH = args.attention_depth
    if _check_args_without_eval('spatial_mask_ratio'):
        config.DATA.SPATIAL_MASK_RATIO = args.spatial_mask_ratio
    if _check_args_without_eval("spatial_refactor_eta"):
        config.TRAIN.SPATIAL_RF_ETA = args.spatial_refactor_eta
    if _check_args("spatial_attention_depth"):
        type = config.MODEL.TYPE
        if type == "SPECTRAL_FORMER":
            # config.MODEL.eval(config.MODEL.TYPE).DEPTH = args.attention_depth
            config.MODEL.SPECTRAL_FORMER.SPATIAL_DEPTH = args.spatial_attention_depth
        elif type == "Dtransformer":
            config.MODEL.Dtransformer.SPATIAL_DEPTH = args.spatial_attention_depth
        elif type == 'swin':
            pass
        else:
            raise Exception(f"not support type:{type}")


    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
