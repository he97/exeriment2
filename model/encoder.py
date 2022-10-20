import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from model.vision_transformer import VisionTransformer


class VisionTransformerForDemo(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        if mask is None:
            x = self.patch_embed(x)

            B, L, _ = x.shape

            # mask_token = self.mask_token.expand(B, L, -1)
            # w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            # x = x * (1 - w) + mask_token * w

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

            if self.pos_embed is not None:
                x = x + self.pos_embed
            x = self.pos_drop(x)

            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            for blk in self.blocks:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            x = self.norm(x)

            x = x[:, 0,:]
            return x
        else:
            x = self.patch_embed(x)

            assert mask is not None
            B, L, _ = x.shape

            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

            if self.pos_embed is not None:
                x = x + self.pos_embed
            x = self.pos_drop(x)

            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            for blk in self.blocks:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            x = self.norm(x)

            x = x[:, 1:]
            return x
