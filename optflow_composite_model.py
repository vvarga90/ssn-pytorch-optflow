import torch
import torch.nn as nn

from lib.ssn.ssn import ssn_iter, sparse_ssn_iter
from base_model import FeatureExtractionModel


class SSNModelCompositeOptflow(nn.Module):

    def __init__(self, color_feature_dim, optflow_feature_dim, color_weights_path=None, n_iter=10):
        '''
        Parameters:
            color_feature_dim: int; the deep feature vector estimated by the pretrained LABYX model 
                                                            (excluding the concatenated LABYX channels)
            optflow_feature_dim: int; the deep feature vector estimated by the optical flow model
            color_weights_path: str; the path to the pretrained LABYX model weights; 
                        if the whole SSNModelCompositeOptflow model weights are loaded from file, no need to pass this
            n_iter: int
        '''
        super().__init__()
        self.n_iter = n_iter
        self.color_feature_dim = color_feature_dim
        self.feature_extract_color = FeatureExtractionModel(deep_feature_dim=color_feature_dim, n_in_channels=5, \
                                                                n_mid_channels=64, concat_in_channels_to_out=True)
        if color_weights_path is not None:
            color_weights_state_dict = torch.load(color_weights_path)

            # modify weight key names, TODO solve it in a better way later
            color_weights_state_dict_mod = {}
            for k, v in color_weights_state_dict.items():
                assert k[:16] == 'feature_extract.'
                color_weights_state_dict_mod[k[16:]] = v
            # END TODO

            self.feature_extract_color.load_state_dict(color_weights_state_dict_mod)

        self.optflow_feature_dim = optflow_feature_dim
        self.feature_extract_optflow = FeatureExtractionModel(deep_feature_dim=optflow_feature_dim, \
                                            n_in_channels=4, n_mid_channels=32, concat_in_channels_to_out=False)
        
    def get_base_feature_dim(self):
        return self.color_feature_dim + 5   # including LABYX channels attached to the output of the base feature extractor
        
    def get_composite_feature_dim(self):
        return self.color_feature_dim + 5 + self.optflow_feature_dim

    def forward(self, x_labyx, x_optflow, nspix, pos_feature_idx_offset=None, pos_scale_dist=None, feature_extraction_only=False,\
                    return_translation_invariant_features=False):
        '''
        Parameters:
            x_labyx: T(batch_size, n_ch_pix:[L,A,B,Y,X], sy, sx) of float
            x_optflow: T(batch_size, n_ch_pix, sy, sx) of float
            nspix: int;
            pos_feature_idx_offset: None or int; if given, YX coordinate channel are expected at 
                                        x_labyx[:, pos_feature_idx_offset:pos_feature_idx_offset+2, ...]
            pos_scale_dist: None OR float; YX coordinate channels during distance computation are scaled by this factor.
                                    A value of 1.0 is the default. A higher value induces increased compactness.
            feature_extraction_only: bool; if True: r1, r2, r3 are not computed, instead, None is returned.
            return_translation_invariant_features: bool; 
                In order to use feature maps / feature vectors to describe texture information free of positional information,
                this option must be enabled. However, these features cannot be used to create the SSN segmentation itself.
                Therefore, this must be set to False during training.
        Returns:
            pixel_composite_features: T(batch_size, n_ch_deep_color+n_ch_deep_optflow, sy, sx) of fl32
            r1: torch.Tensor(B, n_SP, H*W) fl32; see ssn.py for details
            r2: torch.Tensor(B, H*W) i64; see ssn.py for details
            r3: torch.Tensor(B, C_comp, n_SP) fl32; see ssn.py for details
        '''
        assert x_labyx.ndim == x_optflow.ndim == 4
        assert x_labyx.shape[0] == x_optflow.shape[0]
        assert x_labyx.shape[2:] == x_optflow.shape[2:]
        assert x_labyx.shape[1] == x_optflow.shape[1] + 1 == 5

        pixel_color_labyx_features = self.feature_extract_color(x_labyx).detach()   # detach: the pretrained color model is not trained here
        pixel_optflow_features = self.feature_extract_optflow(x_optflow)

        pixel_composite_features = torch.cat([pixel_color_labyx_features, pixel_optflow_features], dim=1)   # (B, C_color+5+C_optflow, Y, X)

        if return_translation_invariant_features is True:
            assert not self.training
            x_labyx_transl_inv = x_labyx.clone()
            assert x_labyx_transl_inv.shape[1] == 5
            x_labyx_transl_inv[:,3:,:,:] = 0.    # yx is set to constant in feature extraction input to hide positional information
            pixel_color_labyx_features_transl_inv = self.feature_extract_color(x_labyx_transl_inv).detach()
            pixel_composite_features_transl_inv = torch.cat([pixel_color_labyx_features_transl_inv, pixel_optflow_features], dim=1)   # (B, C_color+5+C_optflow, Y, X)

        if feature_extraction_only is True:
            return pixel_color_labyx_features, pixel_optflow_features, None, None, None

        if self.training:
            r1, r2, r3 = ssn_iter(pixel_composite_features, nspix, self.n_iter)
        else:
            if return_translation_invariant_features is True:
                r1, r2, r3 = sparse_ssn_iter(pixel_composite_features, nspix, self.n_iter, \
                    pos_feature_idx_offset=pos_feature_idx_offset, pos_scale_dist=pos_scale_dist, \
                    pixel_features_translation_invariant=pixel_composite_features_transl_inv)
            else:
                r1, r2, r3 = sparse_ssn_iter(pixel_composite_features, nspix, self.n_iter, \
                    pos_feature_idx_offset=pos_feature_idx_offset, pos_scale_dist=pos_scale_dist)

        if return_translation_invariant_features is True:
            return pixel_composite_features_transl_inv, r1, r2, r3
        else:
            return pixel_composite_features, r1, r2, r3

