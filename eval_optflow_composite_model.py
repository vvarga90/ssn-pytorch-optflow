import os, math
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.meter import Meter
from optflow_composite_model import SSNModelCompositeOptflow
from lib.dataset import bsds, augmentation, davis
from lib.utils.loss import reconstruct_loss_with_cross_entropy, reconstruct_loss_with_mse
from lib.utils.metrics import achievable_segmentation_accuracy, compute_segmentation_iou

from lib.ssn.ssn import sparse_ssn_iter

@torch.no_grad()
def eval_optflow_composite_model(comp_model, loader, color_scale, pos_scale, optflow_scale, nspix, device, \
                            ignore_background=False):

    comp_model_training_mode = comp_model.training
    comp_model.eval()
    sum_asa = 0
    sum_iou = 0
    for data in loader:
        inputs_lab, inputs_optflow, labels = data

        inputs_lab = inputs_lab.to(device)
        inputs_optflow = inputs_optflow.to(device)
        labels = labels.to(device)

        height, width = inputs_lab.shape[-2:]

        nspix_per_axis = int(math.sqrt(nspix)) 
        pos_scale_mod = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)   # INFO: this line was erroneous, now fixed

        coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij'), 0)
        coords = coords[None].repeat(inputs_lab.shape[0], 1, 1, 1).float()

        inputs_labyx = torch.cat([color_scale*inputs_lab, pos_scale_mod*coords], 1)
        inputs_optflow = optflow_scale*inputs_optflow
        _, Q, H, feat = comp_model(x_labyx=inputs_labyx, x_optflow=inputs_optflow, nspix=nspix)

        H = H.reshape(height, width)
        labels = labels.argmax(1).reshape(height, width)
        
        H_npy = H.to("cpu").detach().numpy()
        labels_npy = labels.to("cpu").numpy()
        asa = achievable_segmentation_accuracy(H_npy, labels_npy)
        iou, _ = compute_segmentation_iou(H_npy, labels_npy, ignore_background=ignore_background)
        sum_asa += asa
        sum_iou += iou
    comp_model.train(mode=comp_model_training_mode)   # reset comp model to previous state
    return sum_asa / len(loader), sum_iou / len(loader)

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

    parser.add_argument("--davis2017_root", default="/home/vavsaai/databases/DAVIS/DAVIS2017/", type=str, help="/path/to/DAVIS2017")
    parser.add_argument("--davis_optflow_folder", default="/home/vavsaai/databases/DAVIS/iccv21_TEMP_impl_preprocessed_data/optflow/", \
                                                                                        type=str, help="/path/to/DAVIS_OPTFLOW_FOLDER")
    parser.add_argument("--comp_model_weights", default="./log/best_composite_model.pth", type=str, help="/path/to/weigh.ts")
    parser.add_argument("--deepfdim", default=15, type=int, help="embedding dimension  (!!! excluding LAB,XY,etc. concatenated at the end !!!")
    parser.add_argument("--optflow_deepfdim", default=10, type=int, help="optflow embedding dimension")
    parser.add_argument("--niter", default=5, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=1000, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--optflow_scale", default=5.0, type=float)
    parser.add_argument("--eval_iou_ignore_background", default=True, type=bool)
    args = parser.parse_args()
    print("Config: ", args)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device is:", device)

    comp_model = SSNModelCompositeOptflow(color_feature_dim=args.deepfdim, optflow_feature_dim=args.optflow_deepfdim, \
                                                                        color_weights_path=None, n_iter=args.niter).to("cuda")
    comp_model.load_state_dict(torch.load(args.comp_model_weights))
    comp_model.eval()

    print("Loading data: DAVIS test")
    test_dataset_davis = davis.DAVIS(davis2017_root=args.davis2017_root, split="test", every_n_th_frame=10, \
                                                                    optflow_data_folder=args.davis_optflow_folder)
    test_loader_davis = DataLoader(test_dataset_davis, 1, shuffle=False, drop_last=False)
    print("Loading data done.")

    t0 = time.time()
    davis_asa, davis_iou = eval_optflow_composite_model(comp_model=comp_model, loader=test_loader_davis, color_scale=args.color_scale, \
                                pos_scale=args.pos_scale, optflow_scale=args.optflow_scale, nspix=args.nspix, device=device, \
                                ignore_background=args.eval_iou_ignore_background)
    t1 = time.time()
    print(f"validation asa (DAVIS test 1/10) {davis_asa}, iou {davis_iou}, eval time {t1-t0}")
