# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import sys
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
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, '..', '..', 'deepcluster'))

import paths
import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler
from clustering import preprocess_features
from batch.augmentation.flip_x_axis import flip_x_axis_img
from batch.augmentation.add_noise import add_noise_img
from batch.dataset import DatasetImg
from batch.dataset import DatasetImgUnbal
#############
from batch.dataset import DatasetGrid
from batch.samplers.sampler_test import SampleFull
from batch.samplers.get_all_patches import GetAllPatches
from data.echogram import Echogram
#############
from batch.data_transform_functions.remove_nan_inf import remove_nan_inf_img
from batch.data_transform_functions.db_with_limits import db_with_limits_img
from batch.combine_functions import CombineFunctions
from classifier_linearSVC import SimpleClassifier

def parse_args():
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16', 'vgg16_tweak'], default='vgg16_tweak',
                        help='CNN architecture (default: vgg16)')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=81,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--nmb_category', type=int, default=3,
                        help='number of ground truth classes(category)')
    parser.add_argument('--lr_Adam', default=3e-5, type=float,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_SGD', default=5e-3, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--save_epoch', default=1, type=int,
                        help='save features every epoch number (default: 20)')
    parser.add_argument('--batch', default=32, type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--verbose', type=bool, default=True, help='chatty')
    parser.add_argument('--frequencies', type=list, default=[18, 38, 120, 200],
                        help='4 frequencies [18, 38, 120, 200]')
    parser.add_argument('--window_dim', type=int, default=32,
                        help='window size')
    parser.add_argument('--display_count', type=int, default=100,
                        help='window size')
    parser.add_argument('--sampler_probs', type=list, default=None,
                        help='[bg, sh27, sbsh27, sh01, sbsh01], default=[1, 1, 1, 1, 1]')
    parser.add_argument('--resume',
                        default=os.path.join(current_dir, 'checkpoint.pth.tar'), type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--exp', type=str,
                        default=current_dir, help='path to exp folder')
    parser.add_argument('--optimizer', type=str, metavar='OPTIM',
                        choices=['Adam', 'SGD'], default='Adam', help='optimizer_choice (default: Adam)')
    parser.add_argument('--semi_ratio', type=float, default=0.025, help='ratio of the labeled samples')

    return parser.parse_args(args=[])

def zip_img_label(img_tensors, labels):
    img_label_pair = []
    for i, zips in enumerate(zip(img_tensors, labels)):
        img_label_pair.append(zips)
    print('num_pairs: ', len(img_label_pair))
    return img_label_pair

def flatten_list(nested_list):
    flatten = []
    for list in nested_list:
        flatten.extend(list)
    return flatten


def compute_features(dataloader, model, N, device, args):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    model.eval()
    # discard the label information in the dataloader
    input_tensors = []
    labels = []
    with torch.no_grad():
         for i, (input_tensor, label) in enumerate(dataloader):
            end = time.time()
            input_tensor.double()
            input_var = torch.autograd.Variable(input_tensor.to(device))
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * args.batch: (i + 1) * args.batch] = aux
            else:
                # special treatment for final batch
                features[i * args.batch:] = aux
            input_tensors.append(input_tensor.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            if args.verbose and (i % args.display_count) == 0:
                print('{0} / {1}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      .format(i, len(dataloader), batch_time=batch_time))
         input_tensors = np.concatenate(input_tensors, axis=0)
         labels = np.concatenate(labels, axis=0)
         return features, input_tensors, labels

def sampling_echograms_full(args):
    tr_ratio = [0.97808653, 0.01301181, 0.00890166]
    path_to_echograms = paths.path_to_echograms()

    ########
    samplers_train = torch.load(os.path.join(path_to_echograms, 'sampler3_tr.pt'))
    samplers_bg = torch.load(os.path.join(path_to_echograms, 'train_bg_32766.pt'))

    supervised_count = int(len(samplers_train[0]) * args.semi_ratio)
    total_unsupervised_count = int((len(samplers_train[0]) - supervised_count) * args.nmb_category)
    unlab_size = [int(ratio * total_unsupervised_count) for ratio in tr_ratio]
    if np.sum(unlab_size) != total_unsupervised_count:
        unlab_size[0] += total_unsupervised_count - np.sum(unlab_size)

    samplers_supervised = []
    samplers_unsupervised = []
    for samplers in samplers_train:
        samplers_supervised.append(samplers[:supervised_count])
        samplers_unsupervised.append(samplers[supervised_count:])
    samplers_unsupervised[0].extend(samplers_bg)

    samplers_unbal_unlab = []
    for sampler, size in zip(samplers_unsupervised, unlab_size):
        samplers_unbal_unlab.append(sampler[:size])

    samplers_semi_unbal_unlab_long = []
    for sampler_unb_unl in samplers_unbal_unlab:
        samplers_semi_unbal_unlab_long.extend(sampler_unb_unl)

    num_classes = len(samplers_train)
    list_length = len(samplers_semi_unbal_unlab_long) // num_classes
    samplers_unanno = [samplers_semi_unbal_unlab_long[i*list_length: (i+1)*list_length] for i in range(num_classes)]
    augmentation = CombineFunctions([add_noise_img, flip_x_axis_img])
    data_transform = CombineFunctions([remove_nan_inf_img, db_with_limits_img])

    dataset_unanno = DatasetImg(
        samplers_unanno,
        args.sampler_probs,
        augmentation_function=augmentation,
        data_transform_function=data_transform)

    dataset_anno = DatasetImg(
        samplers_supervised,
        args.sampler_probs,
        augmentation_function=augmentation,
        data_transform_function=data_transform)

    return dataset_unanno, dataset_anno

def sampling_echograms_test(args):
    path_to_echograms = paths.path_to_echograms()
    samplers_test_bal = torch.load(os.path.join(path_to_echograms, 'sampler3_te_bal.pt'))
    samplers_test_unbal = torch.load(os.path.join(path_to_echograms, 'sampler3_te_unbal.pt'))
    data_transform = CombineFunctions([remove_nan_inf_img, db_with_limits_img])

    dataset_test_bal = DatasetImg(
        samplers_test_bal,
        args.sampler_probs,
        augmentation_function=None,
        data_transform_function=data_transform)

    dataset_test_unbal = DatasetImgUnbal(
        samplers_test_unbal,
        args.sampler_probs,
        augmentation_function=None,
        data_transform_function=data_transform)

    return dataset_test_bal, dataset_test_unbal

def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)
    criterion = nn.CrossEntropyLoss()
    cluster_log = Logger(os.path.join(args.exp, 'clusters.pickle'))

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))

    '''
    ##########################################
    ##########################################
    # Model definition
    ##########################################
    ##########################################'''
    model = models.__dict__[args.arch](bn=True, num_cluster=args.nmb_cluster, num_category=args.nmb_category)
    fd = int(model.cluster_layer[0].weight.size()[1])  # due to transpose, fd is input dim of W (in dim, out dim)
    model.cluster_layer = None
    model.category_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model = model.double()
    model.to(device)
    cudnn.benchmark = True

    if args.optimizer is 'Adam':
        print('Adam optimizer: conv')
        optimizer_body = torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr_Adam,
            betas=(0.9, 0.999),
            weight_decay=10 ** args.wd,
        )
    else:
        print('SGD optimizer: conv')
        optimizer_body = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr_SGD,
            momentum=args.momentum,
            weight_decay=10 ** args.wd,
        )
    '''
    ###############
    ###############
    category_layer
    ###############
    ###############
    '''
    model.category_layer = nn.Sequential(
        nn.Linear(fd, args.nmb_category),
        nn.Softmax(dim=1),
    )
    model.category_layer[0].weight.data.normal_(0, 0.01)
    model.category_layer[0].bias.data.zero_()
    model.category_layer = model.category_layer.double()
    model.category_layer.to(device)

    if args.optimizer is 'Adam':
        print('Adam optimizer: conv')
        optimizer_category = torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.category_layer.parameters()),
            lr=args.lr_Adam,
            betas=(0.9, 0.999),
            weight_decay=10 ** args.wd,
        )
    else:
        print('SGD optimizer: conv')
        optimizer_category = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model.category_layer.parameters()),
            lr=args.lr_SGD,
            momentum=args.momentum,
            weight_decay=10 ** args.wd,
        )
    '''
    ########################################
    ########################################
    Create echogram sampling index
    ########################################
    ########################################'''

    print('Sample echograms.')
    dataset_unanno, dataset_anno = sampling_echograms_full(args)
    dataloader_unanno = torch.utils.data.DataLoader(dataset_unanno,
                                                shuffle=False,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                drop_last=False,
                                                pin_memory=True)

    dataloader_anno = torch.utils.data.DataLoader(dataset_anno,
                                                shuffle=False,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                drop_last=False,
                                                pin_memory=True)

    dataset_test_bal, dataset_test_unbal = sampling_echograms_test(args)
    dataloader_test_bal = torch.utils.data.DataLoader(dataset_test_bal,
                                                shuffle=False,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                drop_last=False,
                                                pin_memory=True)

    dataloader_test_unbal = torch.utils.data.DataLoader(dataset_test_unbal,
                                                shuffle=False,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                drop_last=False,
                                                pin_memory=True)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top located layer parameters from checkpoint
            copy_checkpoint_state_dict = checkpoint['state_dict'].copy()
            for key in list(copy_checkpoint_state_dict):
                if 'cluster_layer' in key:
                    del copy_checkpoint_state_dict[key]
            checkpoint['state_dict'] = copy_checkpoint_state_dict
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_body.load_state_dict(checkpoint['optimizer_body'])
            optimizer_category.load_state_dict(checkpoint['optimizer_category'])
            category_save = os.path.join(args.exp,  'category_layer.pth.tar')
            if os.path.isfile(category_save):
                category_layer_param = torch.load(category_save)
                model.category_layer.load_state_dict(category_layer_param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    exp_bal = os.path.join(args.exp, 'bal')
    exp_unbal = os.path.join(args.exp, 'unbal')
    for dir_bal in [exp_bal, exp_unbal]:
        for dir_2 in ['features', 'pca_features', 'pred']:
            dir_to_make = os.path.join(dir_bal, dir_2)
            if not os.path.isdir(dir_to_make):
                os.makedirs(dir_to_make)

    '''
    #######################
    #######################
    MAIN TRAINING
    #######################
    #######################'''
    epoch = 0
    end = time.time()
    print('#####################  Start training at Epoch %d ################'% epoch)
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1]) # remove ReLU at classifier [:-1]
    model.cluster_layer = None
    model.category_layer = None

    '''
    #######################
    #######################
    PSEUDO-LABEL GENERATION
    #######################
    #######################
    '''
    print('Cluster the features')
    features_train_anno, input_tensors_train_anno, labels_train_anno = compute_features(dataloader_anno, model, len(dataset_anno), device=device, args=args)
    train_anno = [features_train_anno, labels_train_anno]
    with open(os.path.join(args.exp, 'train_anno.pickle'), "wb") as f:
        pickle.dump(train_anno, f)

    features_train_unanno, input_tensors_train_unanno, labels_train_unanno = compute_features(dataloader_unanno, model, len(dataset_unanno), device=device, args=args)
    train_unanno = [features_train_unanno, labels_train_unanno]
    with open(os.path.join(args.exp, 'train_unanno.pickle'), "wb") as f:
        pickle.dump(train_unanno, f)
    '''
    TESTSET
    '''
    print('TEST set: Cluster the features')
    features_te_bal, input_tensors_te_bal, labels_te_bal = compute_features(dataloader_test_bal, model, len(dataset_test_bal), device=device, args=args)
    cp_epoch_out_bal = [features_te_bal, labels_te_bal]
    with open(os.path.join(args.exp, 'bal', 'AAA_epoch_%d_te_bal.pickle' % epoch), "wb") as f:
        pickle.dump(cp_epoch_out_bal, f)

    features_te_unbal, input_tensors_te_unbal, labels_te_unbal = compute_features(dataloader_test_unbal, model, len(dataset_test_unbal), device=device, args=args)
    cp_epoch_out_unbal = [features_te_unbal, labels_te_unbal]
    with open(os.path.join(args.exp, 'unbal',  'AAA_epoch_%d_te_unbal.pickle' % epoch), "wb") as f:
        pickle.dump(cp_epoch_out_unbal, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)

