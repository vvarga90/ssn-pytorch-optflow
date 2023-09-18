import math
import numpy as np
import torch
import h5py
import cv2

from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from lib.ssn.ssn import sparse_ssn_iter
from optflow_composite_model import SSNModelCompositeOptflow


def init_optflow_composite_model(color_feature_dim, optflow_feature_dim, n_iter, composite_weights_path):
    """
    Initialize SSNModelCompositeOptflow instance.
    Args:
        color_feature_dim: int
            feature dimension for supervised setting
        n_iter: int
            number of iterations
        composite_weights_path: str
            pretrained weight
    Return:
        comp_model: SSNModelCompositeOptflow instance (torch.nn.Module subclass)
    """
    assert composite_weights_path is not None
    comp_model = SSNModelCompositeOptflow(color_feature_dim=color_feature_dim, optflow_feature_dim=optflow_feature_dim, \
                                                                        color_weights_path=None, n_iter=n_iter).to("cuda")
    comp_model.load_state_dict(torch.load(composite_weights_path))
    comp_model.eval()
    return comp_model


@torch.no_grad()
def inference_optflow_composite_model(comp_model, image_rgb, image_optflow, nspix, color_scale=0.26, pos_scale=2.5, \
                    optflow_scale=1.0, enforce_connectivity=True, return_feature_vecs=False, return_feature_map=False, \
                    base_features_in_return=False, return_translation_invariant_features=False):
    """
    Generate superpixel segmentation.
    Args:
        comp_model: SSNModelCompositeOptflow instance
        image_rgb: numpy.ndarray
            An array of shape (h, w, c)
        image_optflow: numpy.ndarray
            An array of shape (h, w, c=4:[of_fw_y, of_fw_x, of_bw_y, of_bw_x]) of float
        nspix: int
            number of superpixels
        color_scale: float
            color channel factor
        pos_scale: float
            pixel coordinate factor
        optflow_scale: float
            optflow channel factor
        enforce_connectivity: bool
            if True, enforce superpixel connectivity in postprocessing
        return_feature_vecs: bool
            see returns
        return_feature_map: bool
            see returns
        base_features_in_return: bool
            if True, feature maps/vectors of the base model are reutrned (not including LABYX) 
                                                            instead of those estimated by the composite model
        return_translation_invariant_features: bool
            In order to use feature maps / feature vectors to describe texture information free of positional information,
                this option must be enabled. However, these features cannot be used to create the SSN segmentation itself.
                Therefore, this must be set to False during training.
    Return:
        labels: numpy.ndarray
            An array of shape (H, W)
        (OPTIONAL) spix_fvecs: numpy.ndarray
            An array of shape (B=1, C, n_SP); C = C_base if 'base_features_in_return' is True, otherwise C = C_comp
        (OPTIONAL) pix_fmaps_comp: torch.Tensor
            An array of shape (B=1, C, H, W); C = C_base if 'base_features_in_return' is True, otherwise C = C_comp
    """
    height, width = image_rgb.shape[:2]
    assert image_rgb.shape[:2] == image_optflow.shape[:2]
    assert image_rgb.shape[2] + 1 == image_optflow.shape[2] == 4

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)    

    coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda"), indexing='ij'), 0)
    coords = coords[None].float()

    image_lab = rgb2lab(image_rgb)
    image_lab = torch.from_numpy(image_lab).permute(2, 0, 1)[None].to("cuda").float()
    image_optflow = torch.from_numpy(image_optflow).permute(2, 0, 1)[None].to("cuda").float()

    inputs_labyx = torch.cat([color_scale*image_lab, pos_scale*coords], 1)
    inputs_optflow = optflow_scale*image_optflow

    pix_fmaps_comp, Q, labels, spix_fvecs = comp_model(x_labyx=inputs_labyx, x_optflow=inputs_optflow, nspix=nspix, \
                                                        return_translation_invariant_features=return_translation_invariant_features)
                                    # T(B=1, C_comp, H, W), T(B=1, n_SP, H*W), T(B=1, H*W), T(B=1, C_comp, n_SP)
    labels = labels.reshape(height, width).to("cpu").detach().numpy()       # ndarray(H, W)

    if enforce_connectivity is True:
        assert return_feature_vecs is False
        segment_size = height * width / nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = _enforce_label_connectivity_cython(labels[None], min_size, max_size)[0]

    if base_features_in_return is True:
        base_feature_dim = comp_model.get_base_feature_dim()
        pix_fmaps_comp = pix_fmaps_comp[:,:base_feature_dim,:,:]
        spix_fvecs = spix_fvecs[:,:base_feature_dim,:]

    rets = [labels]
    if return_feature_vecs is True:
        spix_fvecs = spix_fvecs.to("cpu").detach().numpy()       # ndarray(B=1, C, n_SP)
        rets.append(spix_fvecs)
    if return_feature_map is True:
        rets.append(pix_fmaps_comp)

    if len(rets) > 1:
        return tuple(rets)
    else:
        return labels


if __name__ == "__main__":
    
    import os
    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="/path/to/image")
    parser.add_argument("--optflow_data", type=str, help="/path/to/optflow_im_data_hdf5")
    parser.add_argument("--comp_model_weights", default="./log/best_composite_model.pth", type=str, help="/path/to/weigh.ts")
    parser.add_argument("--deepfdim", default=15, type=int, help="embedding dimension  (!!! excluding LAB,XY,etc. concatenated at the end !!!")
    parser.add_argument("--optflow_deepfdim", default=10, type=int, help="optflow embedding dimension")
    parser.add_argument("--niter", default=5, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=1000, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--optflow_scale", default=5.0, type=float)
    args = parser.parse_args()

    model = init_optflow_composite_model(color_feature_dim=args.deepfdim, optflow_feature_dim=args.optflow_deepfdim, \
                                                    n_iter=args.niter, composite_weights_path=args.comp_model_weights)
    # load image and optflow data
    image = plt.imread(args.image)
    h5f = h5py.File(args.optflow_data, 'r')
    flow_image = h5f['flow_data']               # (sy, sx, n_optflow_channels) of np.float32
    h5f.close()

    # run composite model inference on image and optflow data
    s = time.time()
    label = inference_optflow_composite_model(comp_model=model, image_rgb=image, image_optflow=flow_image, nspix=args.nspix, \
                    color_scale=args.color_scale, pos_scale=args.pos_scale, optflow_scale=args.optflow_scale, \
                    enforce_connectivity=True, return_feature_vecs=False, return_feature_map=False, \
                    base_features_in_return=False)
    
    print(f"Inference time {time.time() - s}sec")

    out_fpath = os.path.join(IMGS_FOLDER_OUTPATH, im_fname)
    plt.imsave(out_fpath, mark_boundaries(image, label))
