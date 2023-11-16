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
## Training a base model

_**Note:** a trained base model is provided at `./trained_models/best_model.pth`. The model was trained on the BSDS-500 training/validation set._

First, a base model has to be trained on RGB data (BSDS500 by default). The resulting (base) model is an approximate equivalent of the original SSN model.

```
python train_base_model.py --bsds_root /path/to/BSDS500/BSR
```

Optionally, the base model can be validated on the DAVIS 2017 test set during training on BSDS500:

```
python train_base_model.py --bsds_root /path/to/BSDS500/BSR --davis2017_root /path/to/DAVIS2017 --eval_on_davis True
```

## Training a composite model

_**Note:** a trained composite model is provided at `./trained_models/best_composite_model.pth`. The model uses `./trained_models/best_model.pth` as the base model and was further trained on the DAVIS training set._

Once a base model has been trained, a composite model can be trained that relies on the base model and adds an additional feature extraction network that processes additional image channels and runs the SSN iteration over the concatenated base model features and the newly learned features, while training. Currently, additional channels are optical flow channels by default, but the code can be modified easily for other types of input (e.g., depth info). The composite model can be trained the following way:

```
python train_optflow_composite_model.py --bsds_root /path/to/BSDS500/BSR --davis2017_root /path/to/DAVIS2017 --davis_optflow_folder /path/to/DAVIS_OPTICAL_FLOW_HDF5_FOLDER
```

Optical flow estimates must be generated first by some dense optical flow estimation software (e.g., GMA, https://github.com/zacjiang/GMA). The estimates must be put into separate HDF-5 archives for each video. See `train_optflow_composite_model.py` for details on the format. Currently, the training and evaluation scripts for the composite models load optical flow data for the whole DAVIS dataset into memory, which may take up to 40 GB of (main) memory - more memory efficient implementations could possibly exist.

## Evaluation

The trained base model can be evaluated on BSDS500 alone:

```
python eval_base_model.py --bsds_root /path/to/BSDS500/BSR
```

or both BSDS500 and DAVIS 2017:

```
python eval_base_model.py --bsds_root /path/to/BSDS500/BSR --davis2017_root /path/to/DAVIS2017 --eval_on_davis True
```

The trained composite model can be evaluated on DAVIS 2017:

```
python eval_optflow_composite_model.py --davis2017_root /path/to/DAVIS2017 --davis_optflow_folder /path/to/DAVIS_OPTICAL_FLOW_HDF5_FOLDER --eval_on_davis True
```

See the scripts themselves for more information about the command line arguments.

## Inference

A trained base model can be used for inference (superpixel segmentation) based on RGB only:

```
python inference_base_model.py --image davis_pigs_00043.jpg
```

A trained composite model can be used for inference (superpixel segmentation) based on RGB and optical flow estimations. The optical flow estimations are read from a HDF-5 file in the script. See `train_optflow_composite_model.py` for details on the format.

```
python inference_optflow_composite_model.py --image davis_pigs_00043.jpg --comp_model_weights ./log/composite_model_run1801.pth --optflow_data /path/to/DAVIS_OPTICAL_FLOW_HDF5_FOLDER/optflow_gma_pigs.h5 --optflow_data_fr_idx 43
```

Our trained models were used from the `./trained_models/` folder for the demo images.

# Results

DAVIS 2017 validation set, pigs sequence, frame#43

Base model results (nspix=200, color_scale=0.26, pos_scale=10.0), path to model: `./trained_models/best_model.pth`

<img src=https://github.com/vvarga90/ssn-pytorch-optflow/blob/main/results_pigs_base.png width="600">

Composite model results using pre-generated GMA optical flow estimations (nspix=200, color_scale=0.26, pos_scale=10.0, optflow_scale=15.0), path to model: `./trained_models/best_composite_model.pth`

<img src=https://github.com/vvarga90/ssn-pytorch-optflow/blob/main/results_pigs_composite.png  width="600">


