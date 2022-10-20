import torch
from torch import nn
import torch.nn.functional as F

import config
from model.Trans_BCDM_A.net_A import build_Dtransformer

class SimMIMForHsi(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.in_chans = self.encoder.in_chans
        self.groups = self.in_chans // self.encoder_stride

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.encoder.num_features, out_features=self.encoder.patch_size**2*self.encoder.mask_patch_size),
            # nn.Conv1d(
            #     in_channels=self.encoder.in_chans,
            #     out_channels=self.encoder.in_chans, kernel_size=self.encoder.num_features // encoder_stride**2 ,stride=self.encoder.num_features // encoder_stride**2),
            # nn.PixelShuffle(self.encoder_stride),
        )


        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        # B,C,D = z.size()
        # z = z.reshape(-1,D).unsqueeze(1)

        x_rec = self.decoder(z)
        # B,C,D = x_rec.size(
        # x_rec = x_rec.reshape(B,C*self.encoder.mask_patch_size,-1)
        B, C, D = x_rec.size()
        # assert D == self.encoder_stride**2 , '解码后的图形不能转为正常的图像'
        # x_rec = x_rec.reshape((B,C,int(D**0.5),-1))
        mask = mask.unsqueeze(-1).expand(-1, -1, D)
        # mask = mask.repeat_interleave(self.encoder.mask_patch_size, 1).contiguous()
        # B,M= mask.size()
        # mask = mask.reshape((B,M,1,1))
        # mask = mask.expand((B,M,self.encoder_stride,self.encoder_stride))
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.groups
        return loss
    def finetune(self,x):
        return self.encoder.finetune(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'vit':
        raise Exception('vit not support')
        # encoder = VisionTransformerForSimMIM(
        #     img_size=config.DATA.IMG_SIZE,
        #     patch_size=config.MODEL.VIT.PATCH_SIZE,
        #     in_chans=config.MODEL.VIT.IN_CHANS,
        #     num_classes=0,
        #     embed_dim=config.MODEL.VIT.EMBED_DIM,
        #     depth=config.MODEL.VIT.DEPTH,
        #     num_heads=config.MODEL.VIT.NUM_HEADS,
        #     mlp_ratio=config.MODEL.VIT.MLP_RATIO,
        #     qkv_bias=config.MODEL.VIT.QKV_BIAS,
        #     drop_rate=config.MODEL.DROP_RATE,
        #     drop_path_rate=config.MODEL.DROP_PATH_RATE,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #     init_values=config.MODEL.VIT.INIT_VALUES,
        #     use_abs_pos_emb=config.MODEL.VIT.USE_APE,
        #     use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
        #     use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
        #     use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        # encoder_stride = 16
    elif model_type == 'Dtransformer':
        # 首先呢 输入的部分
        encoder = build_Dtransformer(config)
        # encode_stride 是 patch的边长
        encoder_stride = config.DATA.MASK_PATCH_SIZE
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIMForHsi(encoder=encoder, encoder_stride=encoder_stride)
    return model
