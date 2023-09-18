import os, math
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.meter import Meter
from optflow_composite_model import SSNModelCompositeOptflow
from lib.dataset import bsds, augmentation, davis
from lib.utils.loss import reconstruct_loss_with_cross_entropy, reconstruct_loss_with_mse, reconstruct_loss_with_rahman_iou
from lib.utils.metrics import achievable_segmentation_accuracy
from eval_optflow_composite_model import eval_optflow_composite_model

#
# Train a composite model (SSNModelCompositeOptflow) on the DAVIS 2017 dataset
#   a composite model consists of a pretrained (frozen weights) base feature extraction model (input: LABYX)
#       and another, new feature extraction model (input: optflow). The concatenation of the two feature vectors
#       ([deep_fvec_base, LABYX, deep_fvec_optflow]) is input into the SSN iterative mixing algorithm and loss
#       is computed. Only the optflow feature extraction model is trained.
#

def update_param(data, model, optimizer, compactness, color_scale, pos_scale, optflow_scale, nspix, device):

    inputs_lab, inputs_optflow, labels = data

    inputs_lab = inputs_lab.to(device)
    inputs_optflow = inputs_optflow.to(device)
    labels = labels.to(device)

    height, width = inputs_lab.shape[-2:]

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)

    coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij'), 0)
    coords = coords[None].repeat(inputs_lab.shape[0], 1, 1, 1).float()

    inputs_labyx = torch.cat([color_scale*inputs_lab, pos_scale*coords], 1)
    inputs_optflow = optflow_scale*inputs_optflow
    _, Q, H, feat = model(x_labyx=inputs_labyx, x_optflow=inputs_optflow, nspix=nspix)

    recons_loss = reconstruct_loss_with_cross_entropy(Q, labels, normalize_method='linear')
    #recons_loss = reconstruct_loss_with_rahman_iou(Q, labels, normalize_method='linear')
    compact_loss = reconstruct_loss_with_mse(Q, coords.reshape(*coords.shape[:2], -1), H)

    loss = recons_loss + compactness * compact_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item()}


def train(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device is:", device)

    model = SSNModelCompositeOptflow(color_feature_dim=cfg.deepfdim, optflow_feature_dim=cfg.optflow_deepfdim, \
                                            color_weights_path=cfg.base_model_weights, n_iter=cfg.niter).to(device)
    optimizer = optim.Adam(model.parameters(), cfg.lr)

    print("Loading data: DAVIS train")
    augment = augmentation.Compose([augmentation.RandomHorizontalFlip(), augmentation.RandomScale(), augmentation.RandomCrop()])
    train_dataset = davis.DAVIS(davis2017_root=cfg.davis2017_root, split="train", geo_transforms=augment, \
                                                                            optflow_data_folder=cfg.davis_optflow_folder)
    train_loader = DataLoader(train_dataset, cfg.batchsize, shuffle=True, drop_last=True, num_workers=cfg.nworkers)

    print("Loading data: DAVIS test")
    test_dataset_davis = davis.DAVIS(davis2017_root=cfg.davis2017_root, split="test", every_n_th_frame=10, \
                                                                            optflow_data_folder=cfg.davis_optflow_folder)
    test_loader_davis = DataLoader(test_dataset_davis, 1, shuffle=False, drop_last=False)

    print("Loading data done.")
    meter = Meter()

    iterations = 0
    max_val_davis_iou = 0
    while iterations < cfg.train_iter:
        for data in train_loader:
            iterations += 1
            metric = update_param(data=data, model=model, optimizer=optimizer, compactness=cfg.compactness, \
                                    color_scale=cfg.color_scale, pos_scale=cfg.pos_scale, optflow_scale=cfg.optflow_scale, \
                                    nspix=cfg.nspix, device=device)
            meter.add(metric)
            state = meter.state(f"[{iterations}/{cfg.train_iter}]")
            print(state)
            if (iterations % cfg.test_interval) == 0:
                t0 = time.time()
                davis_asa, davis_iou = eval_optflow_composite_model(comp_model=model, loader=test_loader_davis, color_scale=cfg.color_scale, \
                                pos_scale=cfg.pos_scale, optflow_scale=cfg.optflow_scale, nspix=cfg.nspix, device=device)
                t1 = time.time()
                print(f"validation asa (DAVIS test 1/10) {davis_asa}, iou {davis_iou}, eval time {t1-t0}")

                if davis_iou > max_val_davis_iou:
                    max_val_davis_iou = davis_iou
                    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "best_composite_model.pth"))
                    print("    (best_composite_model.pth was overwritten)")

            if iterations == cfg.train_iter:
                break

    unique_id = str(int(time.time()))
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "composite_model"+unique_id+".pth"))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    # Optical flow archives format:
    #       filename within --davis_optflow_folder: <lib/dataset/davis.py:DAVIS.OPTFLOW_H5_FNAME_PREFIX><DAVIS video name>.h5
    #       HDF-5 files with structure:
    #           'flows': HDF-5 Dataset with shape (n_ims-1, sy, sx, 2:[dy, dx]), dtype float16
    #               hdf5_file['flows'][i,:,:,:] should correspond to optical flow estimation of frame #i to frame#i+1
    #           'inv_flows': same; the optical flow estimations over the reversed video
    #               hdf5_file['inv_flows'][i,:,:,:] should correspond to optical flow estimation of frame #i+1 to frame#i
    #                                                                                (original video frame order, not reversed)
    #           values in optical flow estimation are in delta pixels (y,x order) for videos resized to 480x854 (y,x)

    parser.add_argument("--bsds_root", type=str, help="/path/to/BSR")
    parser.add_argument("--davis2017_root", type=str, help="/path/to/DAVIS2017")
    parser.add_argument("--davis_optflow_folder", type=str, help="/path/to/DAVIS_OPTFLOW_FOLDER")
    parser.add_argument("--out_dir", default="./log", type=str, help="/path/to/output directory")
    parser.add_argument("--base_model_weights", default="./log/best_model.pth", type=str, help="/path/to/weigh.ts")
    parser.add_argument("--batchsize", default=6, type=int)
    parser.add_argument("--nworkers", default=8, type=int, help="number of threads for CPU parallel")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--train_iter", default=500000, type=int)
    parser.add_argument("--deepfdim", default=15, type=int, help="embedding dimension (!!! excluding LAB,XY,etc. concatenated at the end !!!")
    parser.add_argument("--optflow_deepfdim", default=10, type=int, help="optflow embedding dimension")
    parser.add_argument("--niter", default=5, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=200, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--optflow_scale", default=20.0, type=float)
    parser.add_argument("--pos_scale", default=5.0, type=float)
    parser.add_argument("--compactness", default=1e-5, type=float)
    parser.add_argument("--test_interval", default=2000, type=int)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Config: ", args)

    train(args)