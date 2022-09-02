from torch import nn

from model.Trans_BCDM_A.net_A import Encoder


class finetune_G(nn.Module):
    def __init__(self, attn_layers, dim):
        super().__init__()
        self.attn_layers = attn_layers
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x, pretrain_model):
        x = pretrain_model.finetune(x)
        x = self.attn_layers(x)
        x = x[:, 0, :]
        x = self.mlp(x)
        return x


def build_finetune_G(config):
    return finetune_G(attn_layers=Encoder(
        dim=config.DATA.PATCH_DIM,
        depth=1,
        heads=2),
        dim=config.DATA.PATCH_DIM)
