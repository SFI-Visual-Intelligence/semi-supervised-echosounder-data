import argparse
import os
import pickle
# import sys
import time
import copy
import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import paths

import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler

from batch.augmentation.flip_x_axis import flip_x_axis
from batch.augmentation.add_noise import add_noise
# from data.echogram import get_echograms
from data.echogram import get_echograms_revised, get_echograms_full
from batch.dataset import Dataset
from batch.dataset import DatasetSampler
from batch.dataset import DatasetVal
from batch.samplers.background import Background
from batch.samplers.seabed import Seabed
from batch.samplers.shool import Shool
from batch.samplers.shool_seabed import ShoolSeabed
from batch.data_transform_functions.remove_nan_inf import remove_nan_inf
from batch.data_transform_functions.db_with_limits import db_with_limits
from batch.label_transform_functions.index_0_1_27 import index_0_1_27
from batch.label_transform_functions.relabel_with_threshold_morph_close import relabel_with_threshold_morph_close
from batch.combine_functions import CombineFunctions
import chang_patch_sampler as cps
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt

def parse_args():
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16', 'vgg16_tweak'], default='vgg16_tweak',
                        help='CNN architecture (default: vgg16)')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=20,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--save_epoch', default=30, type=int,
                        help='save features every epoch number (default: 20)')
    parser.add_argument('--batch', default=16, type=int,
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
    parser.add_argument('--partition', type=str, default='train_only',
                        help='echogram partition (tr/val/te) by year')
    parser.add_argument('--iteration_train', type=int, default=1200,
                        help='num_tr_iterations per one batch and epoch')
    parser.add_argument('--sampler_probs', type=list, default=None,
                        help='[bg, sh27, sbsh27, sh01, sbsh01], default=[1, 1, 1, 1, 1]')
    parser.add_argument('--resume',
                        default=os.path.join(current_dir, 'checkpoint.pth.tar'), type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--exp', type=str,
                        default=current_dir, help='path to exp folder')
    return parser.parse_args(args=[])

args = parse_args()
window_size = [args.window_dim, args.window_dim]
path_to_echograms = paths.path_to_echograms()
with open(os.path.join(path_to_echograms, 'memmap_2014_heave.pkl'), 'rb') as fp:
    eg_names_full = pickle.load(fp)
echograms = get_echograms_full(eg_names_full)
echograms_train, echograms_val, echograms_test = cps.partition_data(echograms, args.partition, portion_train_test=0.8,
                                                                    portion_train_val=0.75)

sampler_bg_train = Background(echograms_train, window_size)
# sampler_sh27_train = Shool(echograms_train, window_size, 27)
# sampler_sh01_train = Shool(echograms_train, window_size, 1)
# sampler_sbsh27_train = ShoolSeabed(echograms_train, window_size, args.window_dim // 4, fish_type=27)
# sampler_sbsh01_train = ShoolSeabed(echograms_train, window_size, args.window_dim // 4, fish_type=1)


data_bg = DatasetSampler(sampler_bg_train, window_size, args.frequencies)
# data_sh27 = DatasetSampler(sampler_sh27_train, window_size, args.frequencies)
# data_sh01 = DatasetSampler(sampler_sh01_train, window_size, args.frequencies)
# data_sbsh27 = DatasetSampler(sampler_sbsh27_train, window_size, args.frequencies)
# data_sbsh01 = DatasetSampler(sampler_sbsh01_train, window_size, args.frequencies)

numpy_bg = []
for i, data in enumerate(data_bg[0]):
    numpy_bg.append(data)
    if ((i+1) % 3000==0):
        torch.save(numpy_bg, 'numpy_bg_%d.pt' %i)
        numpy_bg = []
        print('bg sample no: ',i)
torch.save(numpy_bg, 'numpy_bg_%d.pt' %i)
del numpy_bg


# numpy_sh27 = []
# for i, data in enumerate(data_sh27[0]):
#     numpy_sh27.append(data)
#     if ((i+1) % 3000==0):
#         torch.save(numpy_sh27, 'numpy_sh27_%d.pt' %i)
#         numpy_sh27 = []
#         print('sh27 sample no: ',i)
# torch.save(numpy_sh27, 'numpy_sh27_%d.pt' %i)
# del numpy_sh27
#
# numpy_sh01 = []
# for i, data in enumerate(data_sh01[0]):
#     numpy_sh01.append(data)
#     if ((i+1) % 3000==0):
#         torch.save(numpy_sh01, 'numpy_sh01_%d.pt'%i)
#         numpy_sh01 = []
#         print('sh01 sample no: ',i)
# torch.save(numpy_sh01, 'numpy_sh01_%d.pt'%i)
# del numpy_sh01
#
# numpy_sbsh27 = []
# for i, data in enumerate(data_sbsh27[0]):
#     numpy_sbsh27.append(data)
#     if ((i+1) % 3000==0):
#         torch.save(numpy_sbsh27, 'numpy_sbsh27_%d.pt'%i)
#         numpy_sbsh27 = []
#         print('sbsh27 sample no: ',i)
# torch.save(numpy_sbsh27, 'numpy_sbsh27_%d.pt'%i)
# del numpy_sbsh27
#
# numpy_sbsh01 = []
# for i, data in enumerate(data_sbsh01[0]):
#     numpy_sbsh01.append(data)
#     if ((i+1) % 3000==0):
#         torch.save(numpy_sbsh01, 'numpy_sbsh01_%d.pt'%i)
#         numpy_sbsh01 = []
#         print('sbsh01 sample no: ',i)
# torch.save(numpy_sbsh01, 'numpy_sbsh01_%d.pt'%i)
# del numpy_sbsh01