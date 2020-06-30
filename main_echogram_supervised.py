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
from clustering import preprocess_features
from batch.augmentation.flip_x_axis import flip_x_axis_img
from batch.augmentation.add_noise import add_noise_img
from batch.dataset import DatasetImg
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
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def parse_args():
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16', 'vgg16_tweak'], default='vgg16_tweak',
                        help='CNN architecture (default: vgg16)')
    parser.add_argument('--nmb_category', type=int, default=3,
                        help='number of ground truth classes(category)')
    parser.add_argument('--lr_Adam', default=1e-4, type=float,
                        help='learning rate (default: 0.05)')
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
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--save_epoch', default=30, type=int,
                        help='save features every epoch number (default: 20)')
    parser.add_argument('--batch', default=16, type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--checkpoints', type=int, default=10000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--verbose', type=bool, default=True, help='chatty')
    parser.add_argument('--frequencies', type=list, default=[18, 38, 120, 200],
                        help='4 frequencies [18, 38, 120, 200]')
    parser.add_argument('--window_dim', type=int, default=32,
                        help='window size')
    parser.add_argument('--partition', type=str, default='train_only',
                        help='echogram partition (tr/val/te) by year')
    parser.add_argument('--sampler_probs', type=list, default=None,
                        help='[bg, sh27, sbsh27, sh01, sbsh01], default=[1, 1, 1, 1, 1]')
    parser.add_argument('--resume',
                        default=os.path.join(current_dir, '..', 'checkpoint.pth.tar'), type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--exp', type=str,
                        default=current_dir, help='path to exp folder')
    parser.add_argument('--optimizer', type=str, metavar='OPTIM',
                        choices=['Adam', 'SGD'], default='Adam', help='optimizer_choice (default: Adam)')
    parser.add_argument('--stride', type=int, default=32, help='stride of echogram patches for eval')
    parser.add_argument('--semi_ratio', type=float, default=1, help='ratio of the labeled samples')

    return parser.parse_args(args=[])

def zip_img_label(img_tensors, labels):
    img_label_pair = []
    for i, zips in enumerate(zip(img_tensors, labels)):
        img_label_pair.append(zips)
    print('num_pairs: ', len(img_label_pair))
    return img_label_pair

def sampling_echograms_full(args):
    path_to_echograms = paths.path_to_echograms()
    samplers_train = torch.load(os.path.join(path_to_echograms, 'sampler3_tr.pt'))
    semi_count = int(len(samplers_train[0]) * args.semi_ratio)
    samplers_semi = [samplers[:semi_count] for samplers in samplers_train]
    augmentation = CombineFunctions([add_noise_img, flip_x_axis_img])
    data_transform = CombineFunctions([remove_nan_inf_img, db_with_limits_img])
    dataset_cp = DatasetImg(
        samplers_train,
        args.sampler_probs,
        augmentation_function=augmentation,
        data_transform_function=data_transform)
    dataset_semi = DatasetImg(
        samplers_semi,
        args.sampler_probs,
        augmentation_function=augmentation,
        data_transform_function=data_transform)
    return dataset_cp, dataset_semi

def sampling_echograms_test(args):
    path_to_echograms = paths.path_to_echograms()
    samplers_test = torch.load(os.path.join(path_to_echograms, 'sampler3_te.pt'))
    data_transform = CombineFunctions([remove_nan_inf_img, db_with_limits_img])
    dataset_test = DatasetImg(
        samplers_test,
        args.sampler_probs,
        augmentation_function=None,
        data_transform_function=data_transform)
    return dataset_test

def supervised_train(loader, model, crit, opt, epoch, device, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()

    for i, (input_tensor, label) in enumerate(loader):
        input_var = torch.autograd.Variable(input_tensor.to(device))
        label_var = torch.autograd.Variable(label.to(device,  non_blocking=True))
        output = model(input_var)
        loss = crit(output, label_var.long())

        # record loss
        losses.update(loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 5) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'supervised_Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), loss=losses))
    return losses.avg

def test(dataloader, model, crit, device, args):
    if args.verbose:
        print('Test')
    batch_time = AverageMeter()
    test_losses = AverageMeter()
    end = time.time()
    model.eval()
    with torch.no_grad():
         for i, (input_tensor, label) in enumerate(dataloader):
            input_tensor.double()
            input_var = torch.autograd.Variable(input_tensor.to(device))
            label_var = torch.autograd.Variable(label.to(device))
            output = model(input_var)
            loss = crit(output, label_var.long())
            test_losses.update(loss.item(), input_tensor.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if args.verbose and (i % 10) == 0:
                print('{0} / {1}\t'
                      'TEST_Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(dataloader), loss=test_losses))
         return test_losses.avg

def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))

    model = models.__dict__[args.arch](bn=True, num_cluster=args.nmb_cluster, num_category=args.nmb_category)
    model.cluster_layer = None
    model = model.double()
    model.to(device)
    cudnn.benchmark = True

    if args.optimizer is 'Adam':
        print('Adam optimizer: conv')
        optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr_Adam,
            betas=(0.5, 0.99),
            weight_decay=10 ** args.wd,
        )
    else:
        print('SGD optimizer: conv')
        optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr_SGD,
            momentum=args.momentum,
            weight_decay=10 ** args.wd,
        )

    criterion = nn.CrossEntropyLoss()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            copy_checkpoint_state_dict = checkpoint['state_dict'].copy()
            for key in list(copy_checkpoint_state_dict):
                if 'top_layer' in key:
                    del copy_checkpoint_state_dict[key]
            checkpoint['state_dict'] = copy_checkpoint_state_dict
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp,  '..', 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    # cluster_log = Logger(os.path.join(args.exp,  '..', 'clusters.pickle'))

    # # Create echogram sampling index
    print('Sample echograms.')
    end = time.time()
    _, dataset_semi = sampling_echograms_full(args)
    dataloader_semi = torch.utils.data.DataLoader(dataset_semi,
                                                shuffle=False,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                drop_last=False,
                                                pin_memory=True)

    dataset_test = sampling_echograms_test(args)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                shuffle=False,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                drop_last=False,
                                                pin_memory=True)

    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))
    loss_collect = [[], [], []]

    for epoch in range(args.start_epoch, args.epochs):
        # remove head
        with torch.autograd.set_detect_anomaly(True):
            supervised_loss = supervised_train(dataloader_semi, model, criterion, optimizer, epoch, device, args)

        print('Train time: {0:.2f} s'.format(time.time() - end))
        end = time.time()
        test_loss = test(dataloader_test, model, criterion, device, args)


        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Supervised tr_loss: {2:.3f} \n'
                  'TEST loss: {3:.3f} \n'.format(epoch, time.time() - end, supervised_loss, test_loss))

            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(args.exp,  '..', 'checkpoint.pth.tar'))

        loss_collect[0].append(epoch)
        loss_collect[1].append(supervised_loss)
        loss_collect[2].append(test_loss)

        with open(os.path.join(args.exp, '..', 'loss_collect.pickle'), "wb") as f:
            pickle.dump(loss_collect, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)

