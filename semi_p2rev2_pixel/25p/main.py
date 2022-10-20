# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, '..', '..', 'deepcluster'))

import models
from algorithms_for_comparisonP2 import test_for_comparisonP2_pixel
from samplers_for_comparisonP2 import sampling_echograms_test_for_comparisonP2_pixel


def parse_args():
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    parser.add_argument('--start_epoch', default=450, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16', 'vgg16_tweak'], default='vgg16_tweak',
                        help='CNN architecture (default: vgg16)')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=81,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--nmb_category', type=int, default=3,
                        help='number of ground truth classes(category)')
    parser.add_argument('--n_classes', type=int, default=3,
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
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--pretrain_epoch', type=int, default=0,
                        help='number of pretrain epochs to run (default: 200)')
    parser.add_argument('--save_epoch', default=50, type=int,
                        help='save features every epoch number (default: 20)')
    parser.add_argument('--batch', default=900, type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--pca', default=32, type=int,
                        help='pca dimension (default: 128)')
    parser.add_argument('--checkpoints', type=int, default=50,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--display_count', type=int, default=40,
                        help='display iterations for every <display_count> numbers')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--verbose', type=bool, default=True, help='chatty')
    parser.add_argument('--frequencies', type=list, default=[18, 38, 120, 200],
                        help='4 frequencies [18, 38, 120, 200]')
    parser.add_argument('--window_dim', type=int, default=32,
                        help='window size')
    parser.add_argument('--sampler_probs', type=list, default=None,
                        help='[bg, sh27, sbsh27, sh01, sbsh01], default=[1, 1, 1, 1, 1]')
    parser.add_argument('--resume',
                        default=os.path.join(current_dir, 'checkpoint.pth.tar'), type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--exp', type=str,
                        default=current_dir, help='path to exp folder')
    parser.add_argument('--pred_test', type=str, default=os.path.join(current_dir, 'test', 'pred'), help='path to exp folder')
    parser.add_argument('--pred_2019', type=str, default=os.path.join(current_dir, 'test', '2019'), help='path to exp folder')
    parser.add_argument('--optimizer', type=str, metavar='OPTIM',
                        choices=['Adam', 'SGD'], default='Adam', help='optimizer_choice (default: Adam)')
    parser.add_argument('--semi_ratio', type=float, default=0.25, help='ratio of the labeled samples')
    parser.add_argument('--f1_avg', type=str, default='weighted', help='the way computing f1-score')
    return parser.parse_args(args=[])


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
    model.to(device, dtype=torch.double)
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
    model.category_layer.to(device, dtype=torch.double)

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
    #############################################################################
    ######### For paper 2, by request of the review 2 (revision 2) ##############
    #############################################################################
    '''
    dataset_te = sampling_echograms_test_for_comparisonP2_pixel(stride=1)

    dataloader_test = torch.utils.data.DataLoader(dataset_te,
                                                shuffle=False,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                drop_last=False,
                                                pin_memory=True)


    # resume_path = '/storage/p1_results_for_p2/semi%d' % int(100 * args.semi_ratio)
    resume_path = '/Users/changkyu/Desktop/p1_results_for_p2/semi%d' % int(100 * args.semi_ratio)
    args.resume = os.path.join(resume_path, 'checkpoint.pth.tar')
    category_save = os.path.join(resume_path, 'category_layer.pth.tar')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if torch.cuda.is_available():
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
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
            if os.path.isfile(category_save):
                if torch.cuda.is_available():
                    category_layer_param = torch.load(category_save)
                else:
                    category_layer_param = torch.load(category_save, map_location=torch.device('cpu'))
                model.category_layer.load_state_dict(category_layer_param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    test_pred_pixel = test_for_comparisonP2_pixel(dataloader_test, model, device, args)
    torch.save(test_pred_pixel, 'pred_test_pixel_%d.tar' % int(args.semi_ratio * 100))



if __name__ == '__main__':
    args = parse_args()
    main(args)


