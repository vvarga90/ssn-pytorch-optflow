import torch
import torch.nn as nn

from lib.ssn.ssn import ssn_iter, sparse_ssn_iter

def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )

class FeatureExtractionModel(nn.Module):

    def __init__(self, deep_feature_dim, n_in_channels, n_mid_channels=64, concat_in_channels_to_out=True):
        super().__init__()
        self.deep_feature_dim = deep_feature_dim
        self.n_mid_channels = n_mid_channels
        self.concat_in_channels_to_out = concat_in_channels_to_out

        self.scale1 = nn.Sequential(
            conv_bn_relu(n_in_channels, n_mid_channels),
            conv_bn_relu(n_mid_channels, n_mid_channels)
        )
        self.scale2 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(n_mid_channels, n_mid_channels),
            conv_bn_relu(n_mid_channels, n_mid_channels)
        )
        self.scale3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(n_mid_channels, n_mid_channels),
            conv_bn_relu(n_mid_channels, n_mid_channels)
        )
        n_out_channels = n_mid_channels*3+n_in_channels
        self.output_conv = nn.Sequential(
            nn.Conv2d(n_out_channels, deep_feature_dim, 3, padding=1),
            #nn.ReLU(True),      # disabled by vvarga90: a ReLU at the end would prevent negative values in the feature vectors
                                 #                      also, measured accuracy is improved if disabled
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = nn.functional.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = nn.functional.interpolate(s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([x, s1, s2, s3], 1)
        feat = self.output_conv(cat_feat)

        if self.concat_in_channels_to_out is True:
            return torch.cat([feat, x], 1)
        else:
            return feat


class SSNModel(nn.Module):

    def __init__(self, deep_feature_dim, n_iter=10):
        super().__init__()

        self.n_iter = n_iter
        self.deep_feature_dim = deep_feature_dim
        self.feature_extract = FeatureExtractionModel(deep_feature_dim=deep_feature_dim, n_in_channels=5, \
                                                                n_mid_channels=64, concat_in_channels_to_out=True)
        
    def forward(self, x, nspix, pos_feature_idx_offset=None, pos_scale_dist=None, feature_extraction_only=False):
        '''
        Parameters:
            x: T(batch_size, n_ch_pix:[L,A,B,Y,X], sy, sx) of float
            nspix: int;
            pos_feature_idx_offset: None or int; if given, YX coordinate channel are expected at 
                                        x[:, pos_feature_idx_offset:pos_feature_idx_offset+2, ...]
            pos_scale_dist: None OR float; YX coordinate channels during distance computation are scaled by this factor.
                                    A value of 1.0 is the default. A higher value induces increased compactness.
            feature_extraction_only: bool; if True: r1, r2, r3 are not computed, instead, None is returned.
        Returns:
            pixel_features: T(batch_size, n_ch_sp, sy, sx) of fl32
            r1: torch.Tensor(B, n_SP_out, n_SP_in) or (B, n_SP, H*W) fl32; see ssn.py for details
            r2: torch.Tensor(B, n_SP_in) or (B, H*W) i64; see ssn.py for details
            r3: torch.Tensor(B, C, n_SP_out) or (B, C, n_SP) fl32; see ssn.py for details
        '''
        assert x.ndim == 4
        pixel_features = self.feature_extract(x)

        if feature_extraction_only is True:
            return pixel_features, None, None, None

        if self.training:
            r1, r2, r3 = ssn_iter(pixel_features, nspix, self.n_iter)
        else:
            r1, r2, r3 = sparse_ssn_iter(pixel_features, nspix, self.n_iter, \
                pos_feature_idx_offset=pos_feature_idx_offset, pos_scale_dist=pos_scale_dist)
        return pixel_features, r1, r2, r3

