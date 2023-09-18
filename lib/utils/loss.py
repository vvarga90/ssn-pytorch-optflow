import torch
import torch.nn.functional as tnnF

from .sparse_utils import naive_sparse_bmm, sparse_permute


def sparse_reconstruction(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with the sparse matrix
    NOTE: this function doesn't use it in this project, because may not return correct gradients

    Args:
        assignment: torch.sparse_coo_tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    labels = labels.permute(0, 2, 1).contiguous()

    # matrix product between (n_spixels, n_pixels) and (n_pixels, channels)
    spixel_mean = naive_sparse_bmm(assignment, labels) / (torch.sparse.sum(assignment, 2).to_dense()[..., None] + 1e-16)
    if hard_assignment is None:
        # (B, n_spixels, n_pixels) -> (B, n_pixels, n_spixels)
        permuted_assignment = sparse_permute(assignment, (0, 2, 1))
        # matrix product between (n_pixels, n_spixels) and (n_spixels, channels)
        reconstructed_labels = naive_sparse_bmm(permuted_assignment, spixel_mean)
    else:
        # index sampling
        reconstructed_labels = torch.stack([sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0)
    return reconstructed_labels.permute(0, 2, 1).contiguous()


def reconstruction(assignment, labels, hard_assignment=None):
    """
    reconstruction

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    labels = labels.permute(0, 2, 1).contiguous()

    # matrix product between (n_spixels, n_pixels) and (n_pixels, channels)
    spixel_mean = torch.bmm(assignment, labels) / (assignment.sum(2, keepdim=True) + 1e-16)
    if hard_assignment is None:
        # (B, n_spixels, n_pixels) -> (B, n_pixels, n_spixels)
        permuted_assignment = assignment.permute(0, 2, 1).contiguous()
        # matrix product between (n_pixels, n_spixels) and (n_spixels, channels)
        reconstructed_labels = torch.bmm(permuted_assignment, spixel_mean)
    else:
        # index sampling
        reconstructed_labels = torch.stack([sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0)
    return reconstructed_labels.permute(0, 2, 1).contiguous()


def reconstruct_loss_with_cross_entropy(assignment, labels, hard_assignment=None, normalize_method='linear'):
    """
    reconstruction loss with cross entropy

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    assert normalize_method in ['linear', 'logsoftmax']
    reconstructed_labels = reconstruction(assignment, labels, hard_assignment) 
    if normalize_method == 'linear':
        reconstructed_labels = reconstructed_labels / (1e-16 + reconstructed_labels.sum(1, keepdim=True))
    elif normalize_method == 'logsoftmax':
        reconstructed_labels = tnnF.softmax((reconstructed_labels + 1e-16).log(), dim=1)
    mask = labels > 0
    return -(reconstructed_labels[mask] + 1e-16).log().mean()


def reconstruct_loss_with_mse(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with mse

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    reconstracted_labels = reconstruction(assignment, labels, hard_assignment)
    return torch.nn.functional.mse_loss(reconstracted_labels, labels)


def reconstruct_loss_with_rahman_iou(assignment, labels, hard_assignment=None, normalize_method='linear'):
    """
    Reconstruction loss using a mean-IoU metric approximation from
        Rahman et al. 2016, "Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation"
    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    assert normalize_method in ['linear', 'logsoftmax']
    reconstructed_labels = reconstruction(assignment, labels, hard_assignment)   # (B, C, n_pixels) fl32, in interval [0,1]
    if normalize_method == 'linear':
        reconstructed_labels = reconstructed_labels / (1e-16 + reconstructed_labels.sum(1, keepdim=True))
    elif normalize_method == 'logsoftmax':
        reconstructed_labels = tnnF.softmax((reconstructed_labels + 1e-16).log(), dim=1)

    labels_present_mask = labels.sum(dim=2) > 0.   # (B, C) bool
    n_labels_present_per_sample = torch.count_nonzero(labels_present_mask, dim=1)     # (B,) int
    assert torch.all(n_labels_present_per_sample > 0)

    inter = reconstructed_labels * labels
    union = reconstructed_labels + labels - inter
    inter = inter.sum(dim=2)                # (B, C)
    union = union.sum(dim=2)                # (B, C)
    iou = inter / (union + 1e-8)            # (B, C)
    iou = iou.sum(dim=1) / n_labels_present_per_sample    # (B,)
    return 1. - iou.mean()

