from torch import nn

from model.SSFTTnet import build_SSFTTnet
from model.Trans_BCDM_A.net_A import build_Dtransformer, build_Dtransformer_as_G, Dtransformer_for_spatial, Encoder
from model.classifier import Classifier, FResClassifier, PengResClassifier, AttentionClassifier
from model.decoder import spectral_decoder, spatial_decoder
from model.encoder import VisionTransformerForDemo
from model.finetune_model import build_finetune_G
from model.mix_model import mix_spatial_spectral
from model.pretrain_model import build_model, SimMIMForHsi
import math
from functools import partial

from model.spectral_former_vit_pytorch import build_spectral_former


def get_pretrain_model(config):
    return build_model(config)

def get_finetune_G(config):
    return build_finetune_G(config)


def get_spatial_decoder(config):
    if config.DATA.SPATIAL.PCA:
        out_dim = config.DATA.SPATIAL.PATCH_SIZE ** 2 * config.DATA.SPATIAL.COMPONENT_NUM
    else:
        out_dim = config.DATA.SPATIAL.PATCH_SIZE ** 2 * config.DATA.SPECTRAL.CHANNEL_DIM
    patches_num = math.ceil((config.DATA.SPATIAL.HALF_WIDTH * 2 + 1) / config.DATA.SPATIAL.PATCH_SIZE) ** 2
    patch_dim = config.MODEL.SPATIAL_PATCH_DIM
    return spatial_decoder(patch_dim,out_dim,patches_num)


def get_decoder(config):
    if config.DATA.MODE == 'spectral':
        return get_spectral_decoder(config)
    elif config.DATA.MODE == 'spatial':
        return get_spatial_decoder(config)
    elif config.DATA.MODE=='spatial+spectral':
        return get_spatial_decoder(config), get_spectral_decoder(config)
    else:
        raise Exception(f'{config.DATA.MODE} decoder not support yet')


def get_spectral_decoder(config):
    encoder_stride = config.DATA.MASK_PATCH_SIZE
    model = spectral_decoder(config, encoder_stride=encoder_stride)
    return model


def get_spectral_G(model_type,config):
    if model_type == 'vit':
        # raise Exception('vit not support')
        encoder = VisionTransformerForDemo(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            group = config.DATA.CHANNEL_DIM // config.DATA.MASK_PATCH_SIZE,
            embed_dim_in=config.DATA.IMG_SIZE**2*config.DATA.MASK_PATCH_SIZE,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        encoder_stride = 16
    elif model_type == 'Dtransformer':
        # 首先呢 输入的部分
        encoder = build_Dtransformer_as_G(config)
        # encode_stride 是 patch的边长
        # encoder_stride = config.DATA.MASK_PATCH_SIZE
    elif model_type == 'SPECTRAL_FORMER':
        encoder = build_spectral_former(config)
    elif model_type == 'SSFTTNET':
        encoder = build_SSFTTnet(config)

    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")
    return encoder
def get_spatial_G(model_type,config):
    if model_type == 'Dtransformer':
        if config.DATA.SPATIAL.PCA:
            in_dim = config.DATA.SPATIAL.PATCH_SIZE**2*config.DATA.SPATIAL.COMPONENT_NUM
        else:
            in_dim = config.DATA.SPATIAL.PATCH_SIZE ** 2 * config.DATA.SPECTRAL.CHANNEL_DIM
        in_channels = math.ceil((config.DATA.SPATIAL.HALF_WIDTH*2+1)/config.DATA.SPATIAL.PATCH_SIZE)**2
        patch_dim = config.MODEL.SPATIAL_PATCH_DIM
        return Dtransformer_for_spatial(
            in_dim = in_dim,
            in_chans=in_channels,
            patch_dim=patch_dim,
            # patch_size=config.MODEL.Dtransformer.PATCH_SIZE,
            attn_layers=Encoder(
                dim=patch_dim,
                depth=config.MODEL.Dtransformer.SPATIAL_DEPTH,
                heads=2)
            )
def get_G(config):
    model_type = config.MODEL.TYPE
    if config.DATA.MODE == 'spectral':
        return get_spectral_G(model_type,config)
    elif config.DATA.MODE == 'spatial':
        return get_spatial_G(model_type,config)
    elif config.DATA.MODE=='spatial+spectral':
        return get_spatial_G(model_type, config),get_spectral_G(model_type, config)
    else:
        raise Exception('this mode not have mode')

def get_mix_model(config):
    '''
    mix two decoder output  use mlp
    in:a decoder out+ b decoder out
    output:classifier:in
    :param config:
    :return:
    '''
    spectral_out_dim = config.MODEL.SPECTRAL_PATCH_DIM
    spatial_out_dim = config.MODEL.SPATIAL_PATCH_DIM
    classifier_in_dim =config.MODEL.CLASSIFIER_IN_DIM
    return mix_spatial_spectral(spectral_out_dim=spectral_out_dim,
                                spatial_out_dim=spatial_out_dim,
                                classifier_in_dim=classifier_in_dim)

def get_classifier(config,depth=3):
    if config.CLASSIFIER.MODE == 'attention':
        return AttentionClassifier(num_classes=config.DATA.CLASS_NUM,
                      in_unit=config.MODEL.SPECTRAL_PATCH_DIM+config.MODEL.SPATIAL_PATCH_DIM,
                      middle=1024,
                      attention=Encoder
                                  (
                                   dim=config.CLASSIFIER.ATTENTION.DIM,
                                   depth=config.CLASSIFIER.ATTENTION.DEPTH,
                                   heads=2),
                      prob=0.2,
                      middle_depth=3)
    return Classifier(num_classes=config.DATA.CLASS_NUM,
                      in_unit=config.MODEL.SPECTRAL_PATCH_DIM+config.MODEL.SPATIAL_PATCH_DIM,
                      middle=1024,
                      prob=0.2,
                      middle_depth=3)
