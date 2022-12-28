import torch
from torch import nn
import torch.nn.functional as F

class spectral_decoder(nn.Module):
    def __init__(self, config,encoder_stride):
        super().__init__()
        self.encoder_stride = encoder_stride
        self.in_chans = config.MODEL.Dtransformer.IN_CHANS
        self.groups = self.in_chans // self.encoder_stride

        self.decoder = nn.Sequential(
            nn.Linear(in_features=config.MODEL.Dtransformer.PATCH_DIM, out_features=config.MODEL.Dtransformer.PATCH_SIZE**2*config.DATA.MASK_PATCH_SIZE),
            # nn.Conv1d(
            #     in_channels=self.encoder.in_chans,
            #     out_channels=self.encoder.in_chans, kernel_size=self.encoder.num_features // encoder_stride**2 ,stride=self.encoder.num_features // encoder_stride**2),
            # nn.PixelShuffle(self.encoder_stride),
        )


        self.patch_size = config.MODEL.Dtransformer.PATCH_SIZE

    def forward(self, x, mask,rec):
        x_rec = self.decoder(rec)

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
    @torch.jit.ignore
    def no_weight_decay(self):
        # if hasattr(self.encoder, 'no_weight_decay'):
        #     return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        # if hasattr(self.encoder, 'no_weight_decay_keywords'):
        #     return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

class spatial_decoder(nn.Module):
    def __init__(self, in_dim,out_dim,patches_num):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.in_dim, out_features=self.out_dim),
            # nn.Conv1d(
            #     in_channels=self.encoder.in_chans,
            #     out_channels=self.encoder.in_chans, kernel_size=self.encoder.num_features // encoder_stride**2 ,stride=self.encoder.num_features // encoder_stride**2),
            # nn.PixelShuffle(self.encoder_stride),
        )
        self.patches_num = patches_num


        # self.patch_size = config.MODEL.Dtransformer.PATCH_SIZE

    def forward(self, x, mask,rec):
        x_rec = self.decoder(rec)

        B, C, D = x_rec.size()
        patches_mask_sum = mask.sum()
        # assert D == self.encoder_stride**2 , '解码后的图形不能转为正常的图像'
        # x_rec = x_rec.reshape((B,C,int(D**0.5),-1))
        mask = mask.unsqueeze(-1).expand(-1, -1, D)
        # mask = mask.repeat_interleave(self.encoder.mask_patch_size, 1).contiguous()
        # B,M= mask.size()
        # mask = mask.reshape((B,M,1,1))
        # mask = mask.expand((B,M,self.encoder_stride,self.encoder_stride))
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)
        return loss
    @torch.jit.ignore
    def no_weight_decay(self):
        # if hasattr(self.encoder, 'no_weight_decay'):
        #     return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        # if hasattr(self.encoder, 'no_weight_decay_keywords'):
        #     return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

class spatial_swin_decoder(nn.Module):
    def __init__(self, in_dim=768,encoder_strider=0, patches_num=0, patch_size=0,dim=0):
        super().__init__()
        assert patch_size!=0, "in decoder,patch_size is 0"
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.encoder_stride = encoder_strider

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_dim,
                out_channels=encoder_strider ** 2 * dim, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )
        self.patches_num = patches_num


        # self.patch_size = config.MODEL.Dtransformer.PATCH_SIZE

    def forward(self, x, mask,rec):
        x_rec = self.decoder(rec)

        B, C, D = x_rec.size()
        patches_mask_sum = mask.sum()
        # assert D == self.encoder_stride**2 , '解码后的图形不能转为正常的图像'
        # x_rec = x_rec.reshape((B,C,int(D**0.5),-1))
        mask = mask.unsqueeze(-1).expand(-1, -1, D)
        # mask = mask.repeat_interleave(self.encoder.mask_patch_size, 1).contiguous()
        # B,M= mask.size()
        # mask = mask.reshape((B,M,1,1))
        # mask = mask.expand((B,M,self.encoder_stride,self.encoder_stride))
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)
        return loss
    @torch.jit.ignore
    def no_weight_decay(self):
        # if hasattr(self.encoder, 'no_weight_decay'):
        #     return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        # if hasattr(self.encoder, 'no_weight_decay_keywords'):
        #     return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}