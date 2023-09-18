
import math
import torch

def calc_init_spixels(pixel_features, n_spixels_y, n_spixels_x):
    """
    Calculate initial superpixels over a pixel grid.
    Args:
        pixel_features: torch.Tensor(B, C, H, W) fl32
        n_spixels_y: int; height of initial superpixel grid
        n_spixels_x: int; width of initial superpixel grid
    Return:
        centroids: torch.Tensor(B, C, n_SP) fl32
        assoc_spix_idxs: torch.Tensor(B, n_SPs_assoc_with_pix=9, H, W) i64; returns the associated superpixel indices for each pixel (own + 8 adjacent initial grid cells)
    """
    batchsize, n_ch, sy, sx = pixel_features.shape
    device = pixel_features.device
    centroids = torch.nn.functional.adaptive_avg_pool2d(pixel_features, (n_spixels_y, n_spixels_x))   # (B, C, n_SPy, n_SPx) fl
    centroids = centroids.reshape(batchsize, n_ch, -1)      # (B, C, n_SPy*n_SPx) fl

    with torch.no_grad():
        blocksize_y_fl = sy / n_spixels_y
        blocksize_x_fl = sx / n_spixels_x
        y_idxs, x_idxs = torch.meshgrid([torch.arange(sy, device=device), torch.arange(sx, device=device)], indexing='ij')   # (H, W), (H, W)
        y_spix_idxs = (y_idxs / blocksize_y_fl).type(torch.int64)               # (H, W) i64
        x_spix_idxs = (x_idxs / blocksize_x_fl).type(torch.int64)               # (H, W) i64
        y_assoc_spix_idxs = torch.tensor([-1, 0, 1], dtype=torch.int64, device=device)[:,None,None] + y_spix_idxs   # (3, H, W) i64
        x_assoc_spix_idxs = torch.tensor([-1, 0, 1], dtype=torch.int64, device=device)[:,None,None] + x_spix_idxs   # (3, H, W) i64
        assoc_spix_idxs = y_assoc_spix_idxs[:,None,:,:] * n_spixels_x + x_assoc_spix_idxs[None,:,:,:]  # (3, 3, H, W) i64
        assoc_spix_idxs = assoc_spix_idxs.reshape(-1, sy, sx)  # (9, H, W) i64
        assoc_spix_idxs = assoc_spix_idxs.expand(batchsize, 9, sy, sx).clone()  # (B, 9, H, W) i64, clone() for safe writing

    return centroids, assoc_spix_idxs

@torch.no_grad()
def sparse_ssn_iter(pixel_features, n_spixels, n_iter, pos_feature_idx_offset=None, pos_scale_dist=1.0, \
                        pixel_features_translation_invariant=None):
    """
    SSN algorithm, sparse implementation only for prediction of superpixels over grid of pixels.
    Args:
        pixel_features: torch.Tensor(B, C, H, W) fl32
        n_spixels: int
        n_iter: int
        pos_feature_idx_offset: None OR int; YX coordinate channel are expected at 
                                    pixel_features[:, pos_feature_idx_offset:pos_feature_idx_offset+2, :]
        pos_scale_dist: float; YX coordinate channels during distance computation are scaled by this factor.
                                    A value of 1.0 is the default. A higher value induces increased compactness.
                                    A low value does not necessarily reduce compactness, 
                                        as the deep pixel features may encode positional information already.
        (OPTIONAL) pixel_features_translation_invariant: torch.Tensor(B, C, H, W) fl32;
                                    If given, 'spixel_features' return is computed from these, while the segmentation 
                                    (sparse_affinity, labels) is computed from 'pixel_features'. This option can be used to 
                                    generate position-invariant feature vectors from position-invariant feature maps.
    Returns:
        sparse_affinity: (SPARSE) torch.Tensor(B, n_SP, H*W) fl32
        labels: torch.Tensor(B, H*W) i64
        spixel_features: torch.Tensor(B, C, n_SP) fl32
    """

    device = pixel_features.device
    batchsize, n_ch, sy, sx = pixel_features.shape
    n_spixels_y = int(math.sqrt(n_spixels * sy / sx))
    n_spixels_x = int(math.sqrt(n_spixels * sx / sy))
    n_spixels = n_spixels_y * n_spixels_x
    assert (pixel_features_translation_invariant is None) or (pixel_features_translation_invariant.shape == pixel_features.shape)

    # INITIALIZE superpixel centroids
    spixel_features, assoc_spix_idxs = calc_init_spixels(pixel_features, n_spixels_y, n_spixels_x)   # (B, C, n_SP), (B, 9, H, W)

    # PRE-COMPUTE all associated PIX <-> SPIX pairs
    assoc_spix_idxs_validmask = (assoc_spix_idxs >= 0) & (assoc_spix_idxs < n_spixels)   # (B, 9, H, W) bool
    assoc_spix_idxs = torch.where(assoc_spix_idxs_validmask, assoc_spix_idxs, torch.tensor([0], dtype=torch.int64, device=device))

    # adjusting positional feature weight (compactness)
    if pos_feature_idx_offset is None:
        pixel_features_w = pixel_features
    else:
        compactness_adjustment_weights = torch.ones((n_ch,), dtype=pixel_features.dtype, device=device)
        compactness_adjustment_weights[pos_feature_idx_offset:pos_feature_idx_offset+2] = pos_scale_dist
        pixel_features_w = pixel_features * compactness_adjustment_weights[None,:,None,None]

    for iter_idx in range(n_iter):

        # COMPUTE DISTANCES for all associated PIX <-> SPIX pairs
        assoc_spix_features = spixel_features[torch.arange(batchsize, device=device)[:,None,None,None], :, assoc_spix_idxs]    # (B, 9, H, W, C) fl -> feature (C) axis goes to back!
        pixel_features_p = pixel_features_w.permute(0, 2, 3, 1)[:,None,:,:,:].expand(batchsize, 9, sy, sx, n_ch)  # (B, 9, H, W, C)

        # adjusting positional feature weight (compactness)
        if pos_feature_idx_offset is not None:
            assoc_spix_features = assoc_spix_features * compactness_adjustment_weights

        dists = torch.nn.functional.pairwise_distance(assoc_spix_features.reshape(-1, n_ch), pixel_features_p.reshape(-1, n_ch))  # (B*9*H*W,)
        dists = dists.reshape(batchsize, 9, sy, sx)   # (B, 9, H, W)
        dists = torch.where(assoc_spix_idxs_validmask, dists, torch.tensor([float("Inf")], device=device))
        affinities = (-dists).softmax(1)         # (B, 9, H, W), affinity_matrix, sum=1 along axis#1
        del dists

        #       place all valid affinities in (batchsize, n_SP, n_pix) sparse tensor
        assoc_spix_idxs_3d = assoc_spix_idxs.reshape(batchsize, 9, sy*sx)                              # (B, 9, H*W)
        assoc_spix_idxs_validmask_3d = assoc_spix_idxs_validmask.reshape(batchsize, 9, sy*sx)       # (B, 9, H*W)
        affinities3d = affinities.reshape(batchsize, 9, sy*sx)                                      # (B, 9, H*W)
        sparse_coords0 = torch.arange(batchsize, device=device)[:,None,None].expand(batchsize, 9, sy*sx)     # (B,) -> (B, 9, H*W)
        sparse_coords2 = torch.arange(sy*sx, device=device).expand(batchsize, 9, sy*sx)                      # (H*W,) -> (B, 9, H*W)
        sparse_coords = torch.stack([sparse_coords0, assoc_spix_idxs_3d, sparse_coords2], dim=0)   # (3:[batch_idx, sp_idx, pix_flat_idx], B, 9, H*W)
        sparse_coords = sparse_coords[:,assoc_spix_idxs_validmask_3d]               # (3, n_valid_sparse_coords)
        affinities3d_valid = affinities3d[assoc_spix_idxs_validmask_3d]                   # (n_valid_sparse_coords,)
        sparse_affinity = torch.sparse_coo_tensor(sparse_coords, affinities3d_valid, size=(batchsize, n_spixels, sy*sx))   # (B, n_SP, H*W)

        # UPDATE SPIXEL FEATURES
        pixel_features_p = pixel_features.reshape(batchsize, n_ch, sy*sx).permute(0, 2, 1)   # (B, H*W, C)
        spixel_features = torch.stack([torch.sparse.mm(m0, m1) for m0, m1 in zip(sparse_affinity, pixel_features_p)], 0)  # (B,) times (n_SP, H*W) X (H*W, C) -> (B, n_SP, C)
        spixel_features = spixel_features / (torch.sparse.sum(sparse_affinity, dim=2).to_dense()[:,:,None] + 1e-16)      # (B, n_SP, C)
        spixel_features = spixel_features.permute(0, 2, 1)      # (B, C, n_SP)

    if pixel_features_translation_invariant is not None:
        pixel_features_transl_inv_p = \
                pixel_features_translation_invariant.reshape(batchsize, n_ch, sy*sx).permute(0, 2, 1)   # (B, H*W, C)
        spixel_features = torch.stack([torch.sparse.mm(m0, m1) for m0, m1 in zip(sparse_affinity, pixel_features_transl_inv_p)], 0)  # (B,) times (n_SP, H*W) X (H*W, C) -> (B, n_SP, C)
        spixel_features = spixel_features / (torch.sparse.sum(sparse_affinity, dim=2).to_dense()[:,:,None] + 1e-16)      # (B, n_SP, C)
        spixel_features = spixel_features.permute(0, 2, 1)      # (B, C, n_SP)

    # get argmax PIX -> SPIX associations for each PIX (hard superpixel labelmap)
    argmax_pix_affinities = affinities3d.argmax(dim=1).reshape(-1)         # (B*H*W)
    assoc_spix_idxs_p = assoc_spix_idxs_3d.permute(0,2,1).reshape(-1, 9)      # (B*H*W, 9)
    labels = assoc_spix_idxs_p[torch.arange(batchsize*sy*sx, device=device), argmax_pix_affinities]      # (B*H*W,)
    labels = labels.reshape(batchsize, sy*sx)      # (B, H*W)

    return sparse_affinity, labels, spixel_features

def ssn_iter(pixel_features, n_spixels, n_iter):
    """
    SSN algorithm, implementation suitable for training.
    Only for pixel grid input.
    The sparse affinity matrix is replaced by a dense matrix with a much greater memory footprint.
    Args:
        pixel_features: torch.Tensor(B, C, H, W) fl32
        n_spixels: int
        n_iter: int
    Returns:
        affinity: torch.Tensor(B, n_SP, H*W) fl32
        labels: torch.Tensor(B, H*W) i64
        spixel_features: torch.Tensor(B, C, n_SP) fl32
    """
    device = pixel_features.device
    batchsize, n_ch, sy, sx = pixel_features.shape
    n_spixels_y = int(math.sqrt(n_spixels * sy / sx))
    n_spixels_x = int(math.sqrt(n_spixels * sx / sy))
    n_spixels = n_spixels_y * n_spixels_x

    # INITIALIZE superpixel centroids
    spixel_features, assoc_spix_idxs = calc_init_spixels(pixel_features, n_spixels_y, n_spixels_x)   # (B, C, n_SP), (B, 9, H, W)

    # PRE-COMPUTE all associated PIX <-> SPIX pairs
    assoc_spix_idxs_validmask = (assoc_spix_idxs >= 0) & (assoc_spix_idxs < n_spixels)   # (B, 9, H, W) bool
    assoc_spix_idxs = torch.where(assoc_spix_idxs_validmask, assoc_spix_idxs, torch.tensor([0], dtype=torch.int64, device=device))

    for iter_idx in range(n_iter):

        # COMPUTE DISTANCES for all associated PIX <-> SPIX pairs
        assoc_spix_features = spixel_features[torch.arange(batchsize, device=device)[:,None,None,None], :, assoc_spix_idxs]    # (B, 9, H, W, C) fl -> feature (C) axis goes to back!
        pixel_features_p = pixel_features.permute(0, 2, 3, 1)[:,None,:,:,:].expand(batchsize, 9, sy, sx, n_ch)  # (B, 9, H, W, C)
        dists = torch.nn.functional.pairwise_distance(assoc_spix_features.reshape(-1, n_ch), pixel_features_p.reshape(-1, n_ch))  # (B*9*H*W,)
        dists = dists.reshape(batchsize, 9, sy, sx)   # (B, 9, H, W)
        #dists.masked_fill_(assoc_spix_idxs_validmask_neg, np.inf)   # filling invalid distances with infinity which will be ignored by softmax (using numpy constant for inf)
        dists = torch.where(assoc_spix_idxs_validmask, dists, torch.tensor([float("Inf")], device=device))
        affinities = (-dists).softmax(1)         # (B, 9, H, W), affinity_matrix, sum=1 along axis#1
        del dists

        #       place all valid affinities in (batchsize, n_SP, n_pix) sparse tensor
        assoc_spix_idxs_3d = assoc_spix_idxs.reshape(batchsize, 9, sy*sx)                              # (B, 9, H*W)
        assoc_spix_idxs_validmask_3d = assoc_spix_idxs_validmask.reshape(batchsize, 9, sy*sx)       # (B, 9, H*W)
        affinities3d = affinities.reshape(batchsize, 9, sy*sx)                                      # (B, 9, H*W)
        sparse_coords0 = torch.arange(batchsize, device=device)[:,None,None].expand(batchsize, 9, sy*sx)     # (B,) -> (B, 9, H*W)
        sparse_coords2 = torch.arange(sy*sx, device=device).expand(batchsize, 9, sy*sx)                      # (H*W,) -> (B, 9, H*W)
        sparse_coords = torch.stack([sparse_coords0, assoc_spix_idxs_3d, sparse_coords2], dim=0)   # (3:[batch_idx, sp_idx, pix_flat_idx], B, 9, H*W)
        sparse_coords = sparse_coords[:,assoc_spix_idxs_validmask_3d]               # (3, n_valid_sparse_coords)
        affinities3d_valid = affinities3d[assoc_spix_idxs_validmask_3d]                   # (n_valid_sparse_coords,)
        sparse_affinity = torch.sparse_coo_tensor(sparse_coords, affinities3d_valid, size=(batchsize, n_spixels, sy*sx))   # (B, n_SP, H*W)
        affinity = sparse_affinity.to_dense()                                       # (B, n_SP, H*W)

        # UPDATE SPIXEL FEATURES
        pixel_features_p = pixel_features.reshape(batchsize, n_ch, sy*sx).permute(0, 2, 1)   # (B, H*W, C)
        spixel_features = torch.bmm(affinity, pixel_features_p)  # (B,) times (n_SP, H*W) X (H*W, C) -> (B, n_SP, C)
        spixel_features = spixel_features / (affinity.sum(dim=2)[:,:,None] + 1e-16)      # (B, n_SP, C)
        spixel_features = spixel_features.permute(0, 2, 1)      # (B, C, n_SP)

    # get argmax PIX -> SPIX associations for each PIX (hard superpixel labelmap)
    argmax_pix_affinities = affinities3d.argmax(dim=1).reshape(-1)         # (B*H*W)
    assoc_spix_idxs_p = assoc_spix_idxs_3d.permute(0,2,1).reshape(-1, 9)      # (B*H*W, 9)
    labels = assoc_spix_idxs_p[torch.arange(batchsize*sy*sx, device=device), argmax_pix_affinities]      # (B*H*W,)
    labels = labels.reshape(batchsize, sy*sx)      # (B, H*W)

    return affinity, labels, spixel_features
