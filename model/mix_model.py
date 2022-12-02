import torch
from torch import nn


class mix_spatial_spectral(nn.Module):
    def __init__(self, spatial_out_dim,spectral_out_dim,classifier_in_dim):
        super().__init__()
        self.spatial_out_dim = spatial_out_dim
        self.spectral_out_dim = spectral_out_dim
        self.classifier_out_dim = classifier_in_dim

        self.decoder = nn.Sequential(
            nn.LayerNorm(self.spatial_out_dim+self.spectral_out_dim),
            nn.Linear(in_features=self.spectral_out_dim+self.spatial_out_dim, out_features=self.classifier_out_dim),
            # nn.Conv1d(
            #     in_channels=self.encoder.in_chans,
            #     out_channels=self.encoder.in_chans, kernel_size=self.encoder.num_features // encoder_stride**2 ,stride=self.encoder.num_features // encoder_stride**2),
            # nn.PixelShuffle(self.encoder_stride),
        )


        # self.patch_size = config.MODEL.Dtransformer.PATCH_SIZE

    def forward(self, spatial_feature,spectral_feature):
        cat_dimension = len(spatial_feature.shape)
        assert spectral_feature.shape[cat_dimension-1] == self.spectral_out_dim and spatial_feature.shape[cat_dimension-1] == self.spatial_out_dim, 'dimension illegal'
        feature = torch.cat((spatial_feature,spectral_feature),cat_dimension-1)
        return feature