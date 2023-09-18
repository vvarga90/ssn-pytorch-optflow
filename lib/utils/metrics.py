
import numpy as np

def achievable_segmentation_accuracy(superpixel, label):
    """
    Function to calculate Achievable Segmentation Accuracy:
        ASA(S,G) = sum_j max_i |s_j \cap g_i| / sum_i |g_i|

    Args:
        input: superpixel image (H, W),
        output: ground-truth (H, W)
    """
    TP = 0
    unique_id = np.unique(superpixel)
    for uid in unique_id:
        mask = superpixel == uid
        label_hist = np.histogram(label[mask])
        maximum_regionsize = label_hist[0].max()
        TP += maximum_regionsize
    return TP / label.size


def compute_segmentation_iou(seg, annot, ignore_background=False, n_labels=None):
    '''
    Parameters:
        seg: ndarray(?) of uint?
        annot: ndarray(?) of uint8
        ignore_background: bool; if True, category with label 0 is not counted in the mean-IoU metric
        (OPTIONAL) n_labels: int; if not specified, computed from the 'annot' array
    Returns:
        iou: float
        n_segments: int
    '''
    assert seg.shape == annot.shape
    ious = []

    if n_labels is None:
        n_labels = np.amax(annot)+1
    assert n_labels >= 2

    # get majority labels for each seg (bg can be majority as well)
    seg_u, seg_inv = np.unique(seg, return_inverse=True)
    seg_inv = seg_inv.reshape(seg.shape)
    n_segs = len(seg_u)
    annot_im_onehot = np.zeros((n_labels,) + annot.shape, dtype=np.bool_)
    seg_label_counts = np.zeros((n_segs, n_labels), dtype=np.int32)   # (n_segs, n_labels)

    for lab in range(0, n_labels):
        annot_im_onehot[lab] = annot == lab
        mseg = seg[annot_im_onehot[lab]]
        mseg_u, mseg_c = np.unique(mseg, return_counts=True)
        assert np.all(np.isin(mseg_u, seg_u))
        mseg_idxs_in_seg = np.searchsorted(seg_u, mseg_u)
        seg_label_counts[mseg_idxs_in_seg, lab] += mseg_c
    assert np.all(np.sum(seg_label_counts, axis=1) > 0)
    majority_labs = np.argmax(seg_label_counts, axis=1)

    # generate rounded annot image
    annot_rounded = majority_labs[seg_inv]

    # compute ious for each (fg) label
    start_lab_idx = 1 if ignore_background is True else 0
    for lab in range(start_lab_idx, n_labels):
        rounded_mask = annot_rounded == lab
        intersection = float(np.count_nonzero(rounded_mask & annot_im_onehot[lab]))
        union = float(np.count_nonzero(rounded_mask | annot_im_onehot[lab]))
        iou = intersection/union if union > 0 else 1.
        ious.append(iou)

    return np.mean(ious), n_segs