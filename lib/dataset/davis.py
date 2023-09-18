import os, glob
import torch
import numpy as np
import scipy.io
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
from PIL import Image    # same as davisinteractive .png loading process ()

import h5py
import cv2

# Script (davis.py) added by vvarga90 based on bsds.py; DAVIS2017 dataset iterator with optical flow support

VIDEO_NAMES = {'train': ['bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus', 'car-turn',\
                           'cat-girl', 'classic-car', 'color-run', 'crossing', 'dance-jump', 'dancing', 'disc-jockey',\
                           'dog-agility', 'dog-gooses', 'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo',\
                           'hike', 'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala', 'lady-running',\
                           'lindy-hop', 'longboard', 'lucia', 'mallard-fly', 'mallard-water', 'miami-surf',\
                           'motocross-bumps', 'motorbike', 'night-race', 'paragliding', 'planes-water', 'rallye',\
                           'rhino', 'rollerblade', 'schoolgirls', 'scooter-board', 'scooter-gray'],
                 'val': ['sheep',\
                        'skate-park', 'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',\
                        'tractor-sand', 'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking'],
                'test': ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout',\
                         'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight',\
                         'goat', 'gold-fish', 'horsejump-high', 'india', 'judo', 'kite-surf', 'lab-coat', 'libby',\
                         'loading', 'mbike-trick', 'motocross-jump', 'paragliding-launch', 'parkour', 'pigs',\
                         'scooter-black', 'shooting', 'soapbox'],
                'train_small': ['bear', 'bmx-bumps'],
                'val_small': ['sheep', 'skate-park'],
                'test_small': ['bike-packing', 'blackswan']}


def convert_label(label):

    onehot = np.zeros((1, 50, label.shape[0], label.shape[1])).astype(np.float32)

    ct = 0
    for t in np.unique(label).tolist():
        if ct >= 50:
            break
        else:
            onehot[:, ct, :, :] = (label == t)
        ct = ct + 1

    return onehot


class DAVIS:

    OPTFLOW_H5_FNAME_PREFIX = 'optflow_gma_'

    CUSTOM_IMSIZE_DICT = {'bike-packing': (480, 910), 'disc-jockey': (480, 1138), 'cat-girl': (480, 911), 'shooting': (480, 1152)}


    def __init__(self, davis2017_root, split, optflow_data_folder=None, color_transforms=None, \
                                                        geo_transforms=None, every_n_th_frame=1):
        # davis2017_root: path should be .../DAVIS/DAVIS2017/
        assert split in ['train', 'val', 'test', 'train_small', 'val_small', 'test_small']
        assert int(every_n_th_frame) == every_n_th_frame
        self.davis2017_root = davis2017_root
        self.optflow_data_folder = optflow_data_folder
        self.split_name = split
        self.color_transforms = color_transforms
        self.geo_transforms = geo_transforms
        self.vidnames = VIDEO_NAMES[self.split_name]

        self.imgs_rgb = []
        self.optflows_fw = []
        self.optflows_bw = []
        self.true_annots = []
        for vidname in self.vidnames:
            img_arr = self._get_img_data(vidname)
            self.imgs_rgb.append(img_arr)
            true_annot_arr = self._get_true_annots(vidname)
            self.true_annots.append(true_annot_arr)

            if self.optflow_data_folder is not None:
                optflow_fw_arr, optflow_bw_arr = self._get_optflow_data(vidname)
                optflow_fw_arr = np.pad(optflow_fw_arr, ((0,1), (0,0), (0,0), (0,0)))
                optflow_bw_arr = np.pad(optflow_bw_arr, ((1,0), (0,0), (0,0), (0,0)))
                self.optflows_fw.append(optflow_fw_arr)
                self.optflows_bw.append(optflow_bw_arr)

        self.imgs_rgb = np.concatenate(self.imgs_rgb, axis=0)                 # nd(n_imgs, sy, sx, 3:rgb) of ui8
        self.true_annots = np.concatenate(self.true_annots, axis=0)           # nd(n_imgs, sy, sx) of ui8
        assert self.imgs_rgb.shape[0] == self.true_annots.shape[0]

        if every_n_th_frame > 1:
            self.imgs_rgb = self.imgs_rgb[::every_n_th_frame].copy()
            self.true_annots = self.true_annots[::every_n_th_frame].copy()  # copy, to make sure, the original (and partly unneded) data is released from memory

        if self.optflow_data_folder is not None:
            self.optflows_fw = np.concatenate(self.optflows_fw, axis=0)       # nd(n_imgs, sy, sx, 2:yx) of fl32
            self.optflows_bw = np.concatenate(self.optflows_bw, axis=0)       # nd(n_imgs, sy, sx, 2:yx) of fl32
            if every_n_th_frame > 1:
                self.optflows_fw = self.optflows_fw[::every_n_th_frame].copy()
                self.optflows_bw = self.optflows_bw[::every_n_th_frame].copy()

            assert self.imgs_rgb.shape[0] == self.optflows_fw.shape[0] == self.optflows_bw.shape[0]
        #


    def __getitem__(self, idx):
        '''
        Returns:
            im_lab: T(3:lab, sy_crop, sx_crop) of fl32; compatible with bsds.py:__getitem__()
            im_of: T(4:n_optflow_channels, sy_crop, sx_crop) of fl32; not present in bsds.py:__getitem__()
                        by default, optflow channels are [fw_dy, fw_dx, bw_dy, bw_dx], 
                            but self.optflow_transform is applied on the optical flow data if given
            gt_annot: T(50, sy_crop*sx_crop) of fl32; compatible with bsds.py:__getitem__()
        '''
        im_lab = rgb2lab(self.imgs_rgb[idx]).astype(np.float32)       # (sy, sx, 3:lab) of fl32
        gt_annot = self.true_annots[idx].astype(np.int64)             # (sy, sx) of i64

        if self.color_transforms is not None:
            im_lab = self.color_transforms(im_lab)
        
        if self.optflow_data_folder is not None:
            im_of_fw = self.optflows_fw[idx].astype(np.float32)           # (sy, sx, 2:yx) of fl32
            im_of_bw = self.optflows_bw[idx].astype(np.float32)           # (sy, sx, 2:yx) of fl32

            if self.geo_transforms is not None:
                im_lab, im_of_fw, im_of_bw, gt_annot = self.geo_transforms([im_lab, im_of_fw, im_of_bw, gt_annot])

            gt_annot = convert_label(gt_annot)
            gt_annot = torch.from_numpy(gt_annot).reshape(50, -1).to(torch.float32) # T(50, sy_crop*sx_crop) of fl32
            im_lab = torch.from_numpy(im_lab).permute(2, 0, 1)                      # T(3:lab, sy_crop, sx_crop) of fl32
            im_of_fw = torch.from_numpy(im_of_fw).permute(2, 0, 1)                  # T(2:yx, sy_crop, sx_crop) of fl32
            im_of_bw = torch.from_numpy(im_of_bw).permute(2, 0, 1)                  # T(2:yx, sy_crop, sx_crop) of fl32
            im_of = torch.cat([im_of_fw, im_of_bw], dim=0)                          # T(n_optflow_total_ch, sy_crop, sx_crop) of fl32
            return im_lab, im_of, gt_annot

        else:
            if self.geo_transforms is not None:
                im_lab, gt_annot = self.geo_transforms([im_lab, gt_annot])

            gt_annot = convert_label(gt_annot)
            gt_annot = torch.from_numpy(gt_annot).reshape(50, -1).to(torch.float32) # T(50, sy_crop*sx_crop) of fl32
            im_lab = torch.from_numpy(im_lab).permute(2, 0, 1)                      # T(3:lab, sy_crop, sx_crop) of fl32
            return im_lab, gt_annot


    def __len__(self):
        return self.imgs_rgb.shape[0]

    def _get_img_data(self, vidname):
        '''
        Loading images with matplotlib to follow original code.
        Parameters:
            vidname: str
        Returns:
            ims_rgb: ndarray(n_frames, sy, sx, 3) of float32
        '''
        vid_folder = os.path.join(self.davis2017_root, 'DAVIS/JPEGImages/480p/', vidname)
        n_frames = len(os.listdir(vid_folder))
        ims_rgb = []
        for fr_idx in range(n_frames):
            im_path = os.path.join(vid_folder, str(fr_idx).zfill(5) + '.jpg')
            im = cv2.imread(im_path, cv2.IMREAD_COLOR)
            im_rgb = plt.imread(im_path)
            if im_rgb.shape != (480,854,3):
                assert vidname in ['disc-jockey', 'bike-packing', 'shooting', 'cat-girl']   # safety check
                im_rgb = cv2.resize(im_rgb, (854,480), interpolation=cv2.INTER_NEAREST)
            ims_rgb.append(im_rgb)
        ims_rgb = np.stack(ims_rgb, axis=0)    # (n_frames, sy, sx, 3) of ui8
        assert ims_rgb.dtype == np.uint8
        return ims_rgb

    def _get_optflow_data(self, vidname):
        '''
        Parameters:
            vidname: str
        Returns:
            flow_fw, flow_bw: ndarray(n_ims-1, sy, sx, 2:[dy, dx]) of fl32
        '''
        assert self.optflow_data_folder is not None
        flows_h5_path = os.path.join(self.optflow_data_folder, DAVIS.OPTFLOW_H5_FNAME_PREFIX + vidname + '.h5')
        h5f = h5py.File(flows_h5_path, 'r')
        flow_fw = h5f['flows'][:].astype(np.float32)
        flow_bw = h5f['inv_flows'][:].astype(np.float32)
        h5f.close()
        return flow_fw, flow_bw

    def _get_true_annots(self, vidname):
        '''
        Parameters:
            vidname: str
        Returns:
            gt_annot: ndarray(n_ims, sy, sx) of uint8
        '''
        davis_annot_folder = os.path.join(self.davis2017_root, 'DAVIS/Annotations/480p', vidname)
        n_frames = len(os.listdir(davis_annot_folder))
        gt_annot = []
        for fr_idx in range(n_frames):
            im_fpath = os.path.join(davis_annot_folder, str(fr_idx).zfill(5) + '.png')
            im = Image.open(im_fpath)   # (sy, sx) of ui8, colors are correctly coded as arange(n_labels) values
            if im.size != (854, 480):  # PIL Image.size attribute: x,y order
                assert vidname in DAVIS.CUSTOM_IMSIZE_DICT.keys()    # safety check
                im = im.resize((854, 480), Image.NEAREST)
            im = np.array(im)
            gt_annot.append(im)
        gt_annot = np.stack(gt_annot, axis=0)    # (n_frames, sy, sx,) of ui8
        assert gt_annot.dtype == np.uint8
        return gt_annot
