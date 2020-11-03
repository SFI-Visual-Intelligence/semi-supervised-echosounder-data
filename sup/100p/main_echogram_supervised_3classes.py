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

from pytorchtools import EarlyStopping
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
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--pretrain_epoch', type=int, default=1000,
                        help='number of pretrain epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--save_epoch', default=100, type=int,
                        help='save features every epoch number (default: 20)')
    parser.add_argument('--batch', default=32, type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--pca', default=32, type=int,
                        help='pca dimension (default: 128)')
    parser.add_argument('--checkpoints', type=int, default=20,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--display_count', type=int, default=200,
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
    parser.add_argument('--optimizer', type=str, metavar='OPTIM',
                        choices=['Adam', 'SGD'], default='Adam', help='optimizer_choice (default: Adam)')
    parser.add_argument('--patience', type=int, default=100, help='Earlystopping patience')
    parser.add_argument('--semi_ratio', type=float, default=1, help='ratio of the labeled samples')
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

def supervised_train(loader, model, crit, opt_body, opt_category, epoch, device, args):
    #############################################################
    # Supervised learning
    supervised_losses = AverageMeter()
    supervised_output_save = []
    supervised_label_save = []
    for i, (input_tensor, label) in enumerate(loader):
        input_var = torch.autograd.Variable(input_tensor.to(device))
        label_var = torch.autograd.Variable(label.to(device, non_blocking=True))
        output = model(input_var)
        supervised_loss = crit(output, label_var.long())

        # compute gradient and do SGD step
        opt_category.zero_grad()
        opt_body.zero_grad()
        supervised_loss.backward()
        opt_category.step()
        opt_body.step()

        # record loss
        supervised_losses.update(supervised_loss.item(), input_tensor.size(0))

        # Record accuracy
        output = torch.argmax(output, axis=1)
        supervised_output_save.append(output.data.cpu().numpy())
        supervised_label_save.append(label.data.cpu().numpy())

        if args.verbose and (i % args.display_count) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'SUPERVISED__Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), loss=supervised_losses))

    supervised_output_flat = flatten_list(supervised_output_save)
    supervised_label_flat = flatten_list(supervised_label_save)
    supervised_accu_list = [out == lab for (out, lab) in zip(supervised_output_flat, supervised_label_flat)]
    supervised_accuracy = sum(supervised_accu_list) / len(supervised_accu_list)
    return supervised_losses.avg, supervised_accuracy

def test(dataloader, model, crit, device, args):
    if args.verbose:
        print('Test')
    test_losses = AverageMeter()
    model.eval()

    test_output_save = []
    test_label_save = []
    with torch.no_grad():
        for i, (input_tensor, label) in enumerate(dataloader):
            input_var = torch.autograd.Variable(input_tensor.to(device))
            label_var = torch.autograd.Variable(label.to(device))
            output = model(input_var)
            loss = crit(output, label_var.long())
            test_losses.update(loss.item(), input_tensor.size(0))

            output = torch.argmax(output, axis=1)
            test_output_save.append(output.data.cpu().numpy())
            test_label_save.append(label.data.cpu().numpy())

            if args.verbose and (i % args.display_count) == 0:
                print('{0} / {1}\t'
                      'TEST_Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(dataloader), loss=test_losses))

    output_flat = flatten_list(test_output_save)
    label_flat = flatten_list(test_label_save)
    accu_list = [out == lab for (out, lab) in zip(output_flat, label_flat)]
    test_accuracy = sum(accu_list) / len(accu_list)
    return test_losses.avg, test_accuracy, output_flat, label_flat

def sampling_echograms_full(args):
    path_to_echograms = paths.path_to_echograms()

    ########
    samplers_train = torch.load(os.path.join(path_to_echograms, 'sampler3_tr.pt'))
    supervised_count = int(len(samplers_train[0]) * args.semi_ratio)
    samplers_supervised = []
    for samplers in samplers_train:
        samplers_supervised.append(samplers[:supervised_count])

    augmentation = CombineFunctions([add_noise_img, flip_x_axis_img])
    data_transform = CombineFunctions([remove_nan_inf_img, db_with_limits_img])

    dataset_semi = DatasetImg(
        samplers_supervised,
        args.sampler_probs,
        augmentation_function=augmentation,
        data_transform_function=data_transform)

    return dataset_semi

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
    dataset_semi = sampling_echograms_full(args)

    dataloader_semi = torch.utils.data.DataLoader(dataset_semi,
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
                # if 'category_layer' in key:
                #     del copy_checkpoint_state_dict[key]
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

    """
    ############################
    ############################
    # PRETRAIN
    ############################
    ############################
    """

    if args.start_epoch < args.pretrain_epoch:
        if os.path.isfile(os.path.join(args.exp, 'pretrain_loss_collect.pickle')):
            with open(os.path.join(args.exp, 'pretrain_loss_collect.pickle'), "rb") as f:
                pretrain_loss_collect = pickle.load(f)
        else:
            pretrain_loss_collect = [[], [], [], [], [], [], []]
        print('Start pretraining with %d percent of the dataset from epoch %d/(%d)'
              % (int(args.semi_ratio * 100), args.start_epoch, args.pretrain_epoch))
        model.cluster_layer = None

        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        for epoch in range(args.start_epoch, args.pretrain_epoch):
            with torch.autograd.set_detect_anomaly(True):
                pre_loss, pre_accuracy = supervised_train(loader=dataloader_semi,
                                                          model=model,
                                                          crit=criterion,
                                                          opt_body=optimizer_body,
                                                          opt_category=optimizer_category,
                                                          epoch=epoch, device=device, args=args)

            '''
            ##############
            ##############
            # TEST phase
            ##############
            ##############
            '''
            test_loss_bal, test_accuracy_bal, test_pred_bal, test_label_bal = test(dataloader_test_bal, model,
                                                                                   criterion, device, args)
            test_loss_unbal, test_accuracy_unbal, test_pred_unbal, test_label_unbal = test(dataloader_test_unbal, model,
                                                                                           criterion, device, args)

            '''Save prediction of the test set'''
            if (epoch % args.save_epoch == 0):
                with open(os.path.join(args.exp, 'bal', 'pred', 'sup_epoch_%d_te_bal.pickle' % epoch), "wb") as f:
                    pickle.dump([test_pred_bal, test_label_bal], f)
                with open(os.path.join(args.exp, 'unbal', 'pred', 'sup_epoch_%d_te_unbal.pickle' % epoch), "wb") as f:
                    pickle.dump([test_pred_unbal, test_label_unbal], f)

            # print log
            if args.verbose:
                print('###### Epoch [{0}] ###### \n'
                      'PRETRAIN tr_loss: {1:.3f} \n'
                      'TEST loss_bal: {2:.3f} \n'
                      'TEST loss_unbal: {3:.3f} \n'
                      'PRETRAIN tr_accu: {4:.3f} \n'
                      'TEST accu bal: {5:.3f} \n'
                      'TEST accu unbal: {6:.3f} \n'.format(epoch, pre_loss, test_loss_bal, test_loss_unbal, pre_accuracy, test_accuracy_bal, test_accuracy_unbal))
            pretrain_loss_collect[0].append(epoch)
            pretrain_loss_collect[1].append(pre_loss)
            pretrain_loss_collect[2].append(test_loss_bal)
            pretrain_loss_collect[3].append(test_loss_unbal)
            pretrain_loss_collect[4].append(pre_accuracy)
            pretrain_loss_collect[5].append(test_accuracy_bal)
            pretrain_loss_collect[6].append(test_accuracy_unbal)

            torch.save({'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer_body': optimizer_body.state_dict(),
                        'optimizer_category': optimizer_category.state_dict(),
                        },
                       os.path.join(args.exp, 'checkpoint.pth.tar'))
            torch.save(model.category_layer.state_dict(), os.path.join(args.exp, 'category_layer.pth.tar'))

            with open(os.path.join(args.exp, 'pretrain_loss_collect.pickle'), "wb") as f:
                pickle.dump(pretrain_loss_collect, f)

            if (epoch+1) % args.checkpoints == 0:
                path = os.path.join(
                    args.exp,
                    'checkpoints',
                    'checkpoint_' + str(epoch) + '.pth.tar',
                )
                if args.verbose:
                    print('Save checkpoint at: {0}'.format(path))
                torch.save({'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'optimizer_body': optimizer_body.state_dict(),
                            'optimizer_category': optimizer_category.state_dict(),
                            }, path)

            early_stopping(test_accuracy_bal, epoch, args, model, optimizer_body, optimizer_category)
            if early_stopping.early_stop:
                print('Early stopping')
                path = os.path.join(
                    args.exp,
                    'checkpoints',
                    'checkpoint_' + str(epoch) + '_final.pth.tar',
                )
                if args.verbose:
                    print('Save checkpoint at: {0}'.format(path))
                torch.save({'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'optimizer_body': optimizer_body.state_dict(),
                            'optimizer_category': optimizer_category.state_dict(),
                            }, path)
                break



if __name__ == '__main__':
    args = parse_args()
    main(args)

