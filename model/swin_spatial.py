import torch
from einops import rearrange
from torch import nn

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

class Swin_hsi_no_mask(nn.Module):

    def __init__(self,
                 name: str = 'swin_tiny',
                 patch_size: int = 4,
                 num_classes: int = 0,
                 num_bands: int = 0,
                 end_stage: int = None,
                 keep_patch_embed: bool = True):
        assert num_classes > 0 and num_bands > 0
        super(Swin_hsi_no_mask, self).__init__()
        self.end_stage = end_stage
        self.patch_size = patch_size

        self.input_proj = nn.Conv2d(num_bands, 3, 1)

        from model.swin_backbone import config_dict, SwinTransformer
        config = config_dict[name]
        config['strides'] = (patch_size, 2, 2, 2)
        config['patch_size'] = patch_size

        if end_stage is not None:
            assert isinstance(end_stage, int) and 0 < end_stage <= len(config['depths'])

        self.body = SwinTransformer(**config)
        if keep_patch_embed and patch_size != 4:
            self.body.patch_embed.projection = nn.Conv2d(3, 96, 4, patch_size)

        delete_keys = ['patch_embed.projection.weight'] \
            if patch_size != 4 and not keep_patch_embed else None
        print(self.body.init_weights(delete_keys))
        freeze(self.body.stages[0])

        self.avg_pool = nn.AdaptiveMaxPool2d(1)

        token_dim = self.body.stages[end_stage-1 if end_stage is not None else -1].embed_dims
        self.output_proj = nn.Sequential(nn.Linear(token_dim, num_classes))


    def forward(self, x, mask=None):

        x = self.input_proj(x)
        # rearrange(x,)
        xs = self.body(x, self.end_stage)
        token = self.avg_pool(xs[self.end_stage-1]).flatten(1)
        return token

    @torch.no_grad()
    def print_output_sizes(self, sample_size):
        print('----------------------')
        print(f'Input Size : \n\t{sample_size}')
        sample = torch.randn(*sample_size)
        outs = self.body(self.input_proj(sample), self.end_stage)
        print('Output Size :')
        for i, out in enumerate(outs):
            print(f'\t{i}:{out.shape}')
        print('----------------------')