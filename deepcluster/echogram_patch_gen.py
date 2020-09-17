# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
# import sys
import time
import faiss
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import paths

import clustering

from batch.augmentation.flip_x_axis import flip_x_axis
from batch.augmentation.add_noise import add_noise
# from data.echogram import get_echograms
from data.echogram import get_echograms_revised, get_echograms_full
from batch.dataset import Dataset
from batch.dataset import DatasetVal
from batch.samplers.background import Background
from batch.samplers.shool import Shool
from batch.samplers.shool_seabed import ShoolSeabed
from batch.data_transform_functions.remove_nan_inf import remove_nan_inf
from batch.data_transform_functions.db_with_limits import db_with_limits
from batch.label_transform_functions.index_0_1_27 import index_0_1_27
from batch.label_transform_functions.relabel_with_threshold_morph_close import relabel_with_threshold_morph_close
from batch.combine_functions import CombineFunctions
import chang_patch_sampler as cps
from data.echogram import Echogram

def parse_args():
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16', 'vgg16_tweak'], default='vgg16_tweak',
                        help='CNN architecture (default: vgg16)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=20,
                        help='number of cluster for k-means (default: 10000)')
    # parser.add_argument('--nmb_class', type=int, default=5,
    #                     help='number of classes of the top layer (default: 6)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--save_epoch', default=10, type=int,
                        help='save features every epoch number (default: 20)')
    parser.add_argument('--batch', default=32, type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--pca', default=32, type=int,
                        help='pca dimension (default: 16)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--checkpoints', type=int, default=200,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--verbose', type=bool, default=True, help='chatty')
    parser.add_argument('--frequencies', type=list, default=[18, 38, 120, 200],
                        help='4 frequencies [18, 38, 120, 200]')
    parser.add_argument('--window_dim', type=int, default=128,
                        help='window size')
    parser.add_argument('--resample_echogram_epoch', type=int, default=300,
                        help='Resample echograms')
    parser.add_argument('--num_echogram', type=int, default=50,
                        help='the size of sampled echograms')
    parser.add_argument('--partition', type=str, default='train_only',
                        help='echogram partition (tr/val/te) by year')
    parser.add_argument('--iteration_train', type=int, default=50,
                        help='num_tr_iterations per one batch and epoch')
    parser.add_argument('--iteration_val', type=int, default=1,
                        help='num_val_iterations per one batch and  epoch')
    parser.add_argument('--sampler_probs', type=list, default=None,
                        help='[bg, sh27, sbsh27, sh01, sbsh01], default=[1, 1, 1, 1, 1]')
    # parser.add_argument('--iteration_test', type=int, default=100,
    #                     help='num_te_iterations per epoch')
    parser.add_argument('--resume',
                        default=os.path.join(current_dir, 'checkpoint.pth.tar'), type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--exp', type=str,
                        default=current_dir, help='path to exp folder')
    return parser.parse_args(args=[])

args = parse_args()

def sampling_echograms_full(window_size, args):
    path_to_echograms = paths.path_to_echograms()
    with open(os.path.join(path_to_echograms, 'memmap_2014_heave.pkl'), 'rb') as fp:
        eg_names_full = pickle.load(fp)
    echograms = get_echograms_full(eg_names_full)
    echograms_train, echograms_val, echograms_test = cps.partition_data(echograms, args.partition, portion_train_test=0.8, portion_train_val=0.75)

    sampler_bg_train = Background(echograms_train, window_size)
    sampler_sh27_train = Shool(echograms_train, window_size, 27)
    sampler_sbsh27_train = ShoolSeabed(echograms_train, window_size, args.window_dim//4, fish_type=27)
    sampler_sh01_train = Shool(echograms_train, window_size, 1)
    sampler_sbsh01_train = ShoolSeabed(echograms_train, window_size, args.window_dim//4, fish_type=1)

    samplers_train = [sampler_bg_train,
                      sampler_sh27_train, sampler_sbsh27_train,
                      sampler_sh01_train, sampler_sbsh01_train]

    augmentation = CombineFunctions([add_noise, flip_x_axis])
    label_transform = CombineFunctions([index_0_1_27, relabel_with_threshold_morph_close])
    data_transform = CombineFunctions([remove_nan_inf, db_with_limits])

    dataset_train = Dataset(
        samplers_train,
        window_size,
        args.frequencies,
        args.batch * args.iteration_train,
        args.sampler_probs,
        augmentation_function=augmentation,
        label_transform_function=label_transform,
        data_transform_function=data_transform)

    return dataset_train

np.random.seed(args.seed)

# load dataset (initial echograms)
window_size = [args.window_dim, args.window_dim]
sampling_idx_check = os.path.join(args.exp, 'sampling_idx.pkl')
if not os.path.isfile(sampling_idx_check):
    print("No saved echogram sampling index. Start with 0")
    sampling_idx = int(0)
else:
    with open(os.path.join(args.exp, 'sampling_idx.pkl'), 'rb') as eg:
        sampling_idx = pickle.load(eg)


dataset_train, sampling_idx = sampling_echograms(sampling_idx, window_size, args)
