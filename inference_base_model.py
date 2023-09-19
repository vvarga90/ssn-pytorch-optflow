import math
import numpy as np
import torch

from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from lib.ssn.ssn import sparse_ssn_iter


def init_model(deep_feature_dim, n_iter, weight=None):
    """
    Initialize SSNModel instance.
    Args:
        deep_feature_dim: int
            feature dimension for supervised setting
        n_iter: int
            number of iterations
        weight: str
            pretrained weight
    Return:
        model: SSNModel instance (torch.nn.Module subclass)
    """
    if weight is None:
        assert False, "TODO implement"
        # model = lambda data: sparse_ssn_iter(data, nspix, n_iter)
    else:
        from base_model import SSNModel
        model = SSNModel(deep_feature_dim=deep_feature_dim, n_iter=n_iter).to("cuda")
        model.load_state_dict(torch.load(weight))
        model.eval()

    return model


@torch.no_grad()
def inference(model, image_rgb, nspix, color_scale=0.26, pos_scale=2.5, enforce_connectivity=True, \
                                                                return_feature_vecs=False, return_feature_map=False):
    """
    Generate superpixel segmentation.
    Args:
        model: SSNModel instance
        image_rgb: numpy.ndarray
            An array of shape (h, w, c)
        nspix: int
            number of superpixels
        color_scale: float
            color channel factor
        pos_scale: float
            pixel coordinate factor
        enforce_connectivity: bool
            if True, enforce superpixel connectivity in postprocessing
    Return:
        labels: numpy.ndarray
            An array of shape (H, W)
        (OPTIONAL) spix_fvecs: numpy.ndarray
            An array of shape (B=1, C, n_SP)
        (OPTIONAL) pix_fmaps: torch.Tensor
            An array of shape (B=1, C, H, W)
    """
    height, width = image_rgb.shape[:2]

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)    

    coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda"), indexing='ij'), 0)
    coords = coords[None].float()

    image_lab = rgb2lab(image_rgb)
    image_lab = torch.from_numpy(image_lab).permute(2, 0, 1)[None].to("cuda").float()

    inputs = torch.cat([color_scale*image_lab, pos_scale*coords], 1)
    pix_fmaps, Q, labels, spix_fvecs = model(x=inputs, nspix=nspix)   # T(B=1, C, H, W), T(B=1, n_SP, H*W), T(B=1, H*W), T(B=1, C, n_SP)
    labels = labels.reshape(height, width).to("cpu").detach().numpy()       # ndarray(H, W)

    if enforce_connectivity is True:
        assert return_feature_vecs is False
        segment_size = height * width / nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = _enforce_label_connectivity_cython(labels[None], min_size, max_size)[0]

    rets = [labels]
    if return_feature_vecs is True:
        spix_fvecs = spix_fvecs.to("cpu").detach().numpy()       # ndarray(B=1, C, n_SP)
        rets.append(spix_fvecs)
    if return_feature_map is True:
        rets.append(pix_fmaps)

    if len(rets) > 1:
        return tuple(rets)
    else:
        return labels


if __name__ == "__main__":
    
    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="/path/to/image")
    parser.add_argument("--base_model_weights", default="./trained_models/best_model.pth", type=str, help="/path/to/weigh.ts")
    parser.add_argument("--deepfdim", default=15, type=int, help="embedding dimension  (!!! excluding LAB,XY,etc. concatenated at the end !!!")
    parser.add_argument("--niter", default=5, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=200, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=10.0, type=float)
    args = parser.parse_args()

    model = init_model(deep_feature_dim=args.deepfdim, n_iter=args.niter, weight=args.base_model_weights)

    image = plt.imread(args.image)

    s = time.time()
    label = inference(model=model, image_rgb=image, nspix=args.nspix, color_scale=args.color_scale, \
                                                                pos_scale=args.pos_scale, enforce_connectivity=False)
    print(f"time {time.time() - s}sec")
    plt.imsave("results.png", mark_boundaries(image, label))
