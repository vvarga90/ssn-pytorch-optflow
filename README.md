# Superpixel Sampling Networks with additional image channel (e.g., optical flow) support

Pure PyTorch implementation of Superpixel Sampling Networks with support for additional image channels (e.g., optical flow). 

Pure PyTorch implementation of SSN base version: https://github.com/vvarga90/ssn-pytorch

This repo was forked from: https://github.com/perrying/ssn-pytorch

paper: https://arxiv.org/abs/1807.10174 
original code: https://github.com/NVlabs/ssn_superpixels

# Requirements
- PyTorch >= 1.4 (Tested with 1.12)
- NumPy
- scikit-image
- matplotlib
- PIL
- h5py

# Usage
## training a base model

First, a base model has to be trained has to be trained on RGB data (BSDS500 by default). The resulting (base) model is an approximate equivalent of the original SSN model.

python train_base_model.py --bsds_root /path/to/BSDS500/BSR

Optionally, the base model can be validated on the DAVIS 2017 test set during training on BSDS500:

python train_base_model.py --bsds_root /path/to/BSDS500/BSR --davis2017_root /path/to/DAVIS2017 --eval_on_davis True

## training a composite model

Once a base model has been trained, a composite model can be trained that relies on the base model and adds an additional feature extraction network that processes additional image channels and runs the SSN iteration over the concatenated base model features and the newly learned features, while training. Currently, additional channels are optical flow channels by default, but the code can be modified easily for other types of input (e.g., depth info). The composite model can be trained the following way:

python train_optflow_composite_model.py --bsds_root /path/to/BSDS500/BSR --davis2017_root /path/to/DAVIS2017 --davis_optflow_folder /path/to/DAVIS_OPTICAL_FLOW_HDF5_FOLDER

Optical flow estimates must be generated first by some dense optical flow estimation software (e.g., GMA, https://github.com/zacjiang/GMA). The estimates must be put into separate HDF-5 archives for each video. See train_optflow_composite_model.py for details on the format.

## evaluation

TODO

## inference

TODO

# Results TODO
SSN_pix  
<img src=https://github.com/vvarga90/ssn-pytorch/blob/master/SSN_pix_result.png>

SSN_deep  
<img src=https://github.com/vvarga90/ssn-pytorch/blob/master/SSN_deep_result.png>


