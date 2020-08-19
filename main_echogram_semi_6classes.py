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
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=64,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--nmb_category', type=int, default=6,
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
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--pretrain_epoch', type=int, default=0,
                        help='number of pretrain epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--save_epoch', default=50, type=int,
                        help='save features every epoch number (default: 20)')
    parser.add_argument('--batch', default=32, type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--pca', default=32, type=int,
                        help='pca dimension (default: 128)')
    parser.add_argument('--checkpoints', type=int, default=10,
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
    parser.add_argument('--semi_ratio', type=float, default=0.1, help='ratio of the labeled samples')

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

        if args.verbose and (i % 5) == 0:
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

            if args.verbose and (i % 10) == 0:
                print('{0} / {1}\t'
                      'TEST_Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(dataloader), loss=test_losses))

    output_flat = flatten_list(test_output_save)
    label_flat = flatten_list(test_label_save)
    accu_list = [out == lab for (out, lab) in zip(output_flat, label_flat)]
    test_accuracy = sum(accu_list) / len(accu_list)
    return test_losses.avg, test_accuracy

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
            if args.verbose and (i % 10) == 0:
                print('{0} / {1}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      .format(i, len(dataloader), batch_time=batch_time))
         input_tensors = np.concatenate(input_tensors, axis=0)
         labels = np.concatenate(labels, axis=0)
         return features, input_tensors, labels

def semi_train(loader, semi_loader, model, fd, crit, opt_body, opt_category, epoch, device, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    semi_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, ((input_tensor, label), pseudo_target, imgidx) in enumerate(loader):

        input_var = torch.autograd.Variable(input_tensor.to(device))
        pseudo_target_var = torch.autograd.Variable(pseudo_target.to(device,  non_blocking=True))
        output = model(input_var)
        loss = crit(output, pseudo_target_var.long())

        # record loss
        losses.update(loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt_body.zero_grad()
        loss.backward()
        opt_body.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 5) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'PSEUDO_Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time, loss=losses))

    '''SUPERVISION with a few labelled dataset'''
    model.cluster_layer = None
    model.category_layer = nn.Sequential(
        nn.Linear(fd, args.nmb_category),
        nn.Softmax(dim=1),
    )
    model.category_layer[0].weight.data.normal_(0, 0.01)
    model.category_layer[0].bias.data.zero_()
    model.category_layer = model.category_layer.double()
    model.category_layer.to(device)

    category_save = os.path.join(args.exp, '..', 'category_layer.pth.tar')
    if os.path.isfile(category_save):
        category_layer_param = torch.load(category_save)
        model.category_layer.load_state_dict(category_layer_param)

    semi_output_save = []
    semi_label_save = []
    for i, (input_tensor, label) in enumerate(semi_loader):
        input_var = torch.autograd.Variable(input_tensor.to(device))
        label_var = torch.autograd.Variable(label.to(device,  non_blocking=True))

        output = model(input_var)
        semi_loss = crit(output, label_var.long())

        # compute gradient and do SGD step
        opt_category.zero_grad()
        opt_body.zero_grad()
        semi_loss.backward()
        opt_category.step()
        opt_body.step()

        # record loss
        semi_losses.update(semi_loss.item(), input_tensor.size(0))

        # Record accuracy
        output = torch.argmax(output, axis=1)
        semi_output_save.append(output.data.cpu().numpy())
        semi_label_save.append(label.data.cpu().numpy())

        # measure elapsed time
        if args.verbose and (i % 5) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'SEMI_Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(semi_loader), loss=semi_losses))

    semi_output_flat = flatten_list(semi_output_save)
    semi_label_flat = flatten_list(semi_label_save)
    semi_accu_list = [out == lab for (out, lab) in zip(semi_output_flat, semi_label_flat)]
    semi_accuracy = sum(semi_accu_list)/len(semi_accu_list)
    return losses.avg, semi_losses.avg, semi_accuracy

def sampling_echograms_full(args):
    path_to_echograms = paths.path_to_echograms()
    samplers_train = torch.load(os.path.join(path_to_echograms, 'sampler6_tr.pt'))

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
    samplers_test = torch.load(os.path.join(path_to_echograms, 'sampler6_te.pt'))
    data_transform = CombineFunctions([remove_nan_inf_img, db_with_limits_img])

    dataset_test = DatasetImg(
        samplers_test,
        args.sampler_probs,
        augmentation_function=None,
        data_transform_function=data_transform)
    return dataset_test

def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)
    criterion = nn.CrossEntropyLoss()
    cluster_log = Logger(os.path.join(args.exp,  '..', 'clusters.pickle'))

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
    dataset_cp, dataset_semi = sampling_echograms_full(args)
    dataloader_cp = torch.utils.data.DataLoader(dataset_cp,
                                                shuffle=False,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                drop_last=False,
                                                pin_memory=True)

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

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster, args.pca)

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
            category_save = os.path.join(args.exp, '..', 'category_layer.pth.tar')
            if os.path.isfile(category_save):
                category_layer_param = torch.load(category_save)
                model.category_layer.load_state_dict(category_layer_param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, '..', 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    '''
    #######################
    #######################    
    PRETRAIN: commented
    #######################
    #######################'''
    # if args.start_epoch < args.pretrain_epoch:
    #     pretrain_loss_collect = [[], [], [], [], []]
    #     print('Start pretraining with %d percent of the dataset from epoch %d/(%d)'
    #           % (int(args.semi_ratio * 100), args.start_epoch, args.pretrain_epoch))
    #     model.cluster_layer = None
    #
    #     for epoch in range(args.start_epoch, args.pretrain_epoch):
    #         with torch.autograd.set_detect_anomaly(True):
    #             pre_loss, pre_accuracy = supervised_train(loader=dataloader_semi,
    #                                                       model=model,
    #                                                       crit=criterion,
    #                                                       opt_body=optimizer_body,
    #                                                       opt_category=optimizer_category,
    #                                                       epoch=epoch, device=device, args=args)
    #         test_loss, test_accuracy = test(dataloader_test, model, criterion, device, args)
    #
    #         # print log
    #         if args.verbose:
    #             print('###### Epoch [{0}] ###### \n'
    #                   'PRETRAIN tr_loss: {1:.3f} \n'
    #                   'TEST loss: {2:.3f} \n'
    #                   'PRETRAIN tr_accu: {3:.3f} \n'
    #                   'TEST accu: {4:.3f} \n'.format(epoch, pre_loss, test_loss, pre_accuracy, test_accuracy))
    #         pretrain_loss_collect[0].append(epoch)
    #         pretrain_loss_collect[1].append(pre_loss)
    #         pretrain_loss_collect[2].append(test_loss)
    #         pretrain_loss_collect[3].append(pre_accuracy)
    #         pretrain_loss_collect[4].append(test_accuracy)
    #
    #         torch.save({'epoch': epoch + 1,
    #                     'arch': args.arch,
    #                     'state_dict': model.state_dict(),
    #                     'optimizer_body': optimizer_body.state_dict(),
    #                     'optimizer_category': optimizer_category.state_dict(),
    #                     },
    #                    os.path.join(args.exp,  '..', 'checkpoint.pth.tar'))
    #         torch.save(model.category_layer.state_dict(), os.path.join(args.exp,  '..', 'category_layer.pth.tar'))
    #
    #         with open(os.path.join(args.exp, '..', 'pretrain_loss_collect.pickle'), "wb") as f:
    #             pickle.dump(pretrain_loss_collect, f)
    #
    #         if (epoch+1) % args.checkpoints == 0:
    #             path = os.path.join(
    #                 args.exp, '..',
    #                 'checkpoints',
    #                 'checkpoint_' + str(epoch) + '.pth.tar',
    #             )
    #             if args.verbose:
    #                 print('Save checkpoint at: {0}'.format(path))
    #             torch.save({'epoch': epoch + 1,
    #                         'arch': args.arch,
    #                         'state_dict': model.state_dict(),
    #                         'optimizer_body': optimizer_body.state_dict(),
    #                         'optimizer_category': optimizer_category.state_dict(),
    #                         }, path)

    nmi_save = []
    loss_collect = [[], [], [], [], [], [], []]
    '''
    #######################
    #######################
    MAIN TRAINING
    #######################
    #######################'''
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()
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
        features_train, input_tensors_train, labels_train = compute_features(dataloader_cp, model, len(dataset_cp), device=device, args=args)
        clustering_loss, pca_features = deepcluster.cluster(features_train, verbose=args.verbose)

        nan_location = np.isnan(pca_features)
        inf_location = np.isinf(pca_features)
        if (not np.allclose(nan_location, 0)) or (not np.allclose(inf_location, 0)):
            print('PCA: Feature NaN or Inf found. Nan count: ', np.sum(nan_location), ' Inf count: ', np.sum(inf_location))
            print('Skip epoch ', epoch)
            torch.save(pca_features, 'tr_pca_NaN_%d.pth.tar' % epoch)
            torch.save(features_train, 'tr_feature_NaN_%d.pth.tar' % epoch)
            continue

        print('Assign pseudo labels')
        size_cluster = np.zeros(len(deepcluster.images_lists))
        for i,  _list in enumerate(deepcluster.images_lists):
            size_cluster[i] = len(_list)
        print('size in clusters: ', size_cluster)
        img_label_pair_train = zip_img_label(input_tensors_train, labels_train)
        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                  img_label_pair_train)  # Reassigned pseudolabel

        # uniformly sample per target
        sampler_train = UnifLabelSampler(int(len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            sampler=sampler_train,
            pin_memory=True,
        )
        '''
        ####################################################################
        ####################################################################
        TRSNSFORM MODEL FOR SELF-SUPERVISION // SEMI-SUPERVISION
        ####################################################################
        ####################################################################
        '''
        # Recover classifier with ReLU (that is not used in clustering)
        mlp = list(model.classifier.children()) # classifier that ends with linear(512 * 128). No ReLU at the end
        mlp.append(nn.ReLU(inplace=True).to(device))
        model.classifier = nn.Sequential(*mlp)
        model.classifier.to(device)

        '''SELF-SUPERVISION (PSEUDO-LABELS)'''
        model.category_layer = None
        model.cluster_layer = nn.Sequential(
            nn.Linear(fd, args.nmb_cluster),  # nn.Linear(4096, num_cluster),
            nn.Softmax(dim=1),  # should be removed and replaced by ReLU for category_layer
        )
        model.cluster_layer[0].weight.data.normal_(0, 0.01)
        model.cluster_layer[0].bias.data.zero_()
        model.cluster_layer = model.cluster_layer.double()
        model.cluster_layer.to(device)

        ''' train network with clusters as pseudo-labels '''
        with torch.autograd.set_detect_anomaly(True):
            pseudo_loss, semi_loss, semi_accuracy = semi_train(train_dataloader, dataloader_semi, model, fd, criterion,
                                                               optimizer_body, optimizer_category, epoch, device=device, args=args)

        # save checkpoint
        if (epoch + 1) % args.checkpoints == 0:
            path = os.path.join(
                args.exp, '..',
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

        '''
        ##############
        ##############
        # TEST phase
        ##############
        ##############
        '''
        test_loss, test_accuracy = test(dataloader_test, model, criterion, device, args)

        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Pseudo tr_loss: {2:.3f} \n'
                  'SEMI tr_loss: {3:.3f} \n'
                  'TEST loss: {4:.3f} \n'
                  'Clustering loss: {5:.3f} \n'
                  'SEMI accu: {6:.3f} \n'
                  'TEST accu: {7:.3f} \n'
                  .format(epoch, time.time() - end, pseudo_loss, semi_loss,
                          test_loss, clustering_loss, semi_accuracy, test_accuracy))
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                nmi_save.append(nmi)
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
                with open("./nmi_collect.pickle", "wb") as ff:
                    pickle.dump(nmi_save, ff)
            except IndexError:
                pass
            print('####################### \n')

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)

        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer_body': optimizer_body.state_dict(),
                    'optimizer_category': optimizer_category.state_dict(),
                    },
                   os.path.join(args.exp, '..', 'checkpoint.pth.tar'))
        torch.save(model.category_layer.state_dict(), os.path.join(args.exp, '..', 'category_layer.pth.tar'))

        loss_collect[0].append(epoch)
        loss_collect[1].append(pseudo_loss)
        loss_collect[2].append(semi_loss)
        loss_collect[3].append(clustering_loss)
        loss_collect[4].append(test_loss)
        loss_collect[5].append(semi_accuracy)
        loss_collect[6].append(test_accuracy)
        with open(os.path.join(args.exp, '..', 'loss_collect.pickle'), "wb") as f:
            pickle.dump(loss_collect, f)

        '''
        ############################
        ############################
        # PSEUDO-LABEL GEN: Test set
        ############################
        ############################
        '''
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1]) # remove ReLU at classifier [:-1]
        model.cluster_layer = None
        model.category_layer = None

        print('TEST set: Cluster the features')
        features_te, input_tensors_te, labels_te = compute_features(dataloader_test, model, len(dataset_test),
                                                                    device=device, args=args)
        clustering_loss_te, pca_features_te = deepcluster.cluster(features_te, verbose=args.verbose)

        mlp = list(model.classifier.children()) # classifier that ends with linear(512 * 128). No ReLU at the end
        mlp.append(nn.ReLU(inplace=True).to(device))
        model.classifier = nn.Sequential(*mlp)
        model.classifier.to(device)

        nan_location = np.isnan(pca_features_te)
        inf_location = np.isinf(pca_features_te)
        if (not np.allclose(nan_location, 0)) or (not np.allclose(inf_location, 0)):
            print('PCA: Feature NaN or Inf found. Nan count: ', np.sum(nan_location), ' Inf count: ',
                  np.sum(inf_location))
            print('Skip epoch ', epoch)
            torch.save(pca_features_te, 'te_pca_NaN_%d.pth.tar' % epoch)
            torch.save(features_te, 'te_feature_NaN_%d.pth.tar' % epoch)
            continue

        # save patches per epochs
        cp_epoch_out = [features_te, deepcluster.images_lists, deepcluster.images_dist_lists, input_tensors_te,
                        labels_te]

        if (epoch % args.save_epoch == 0):
            with open(os.path.join(args.exp, '..', 'cp_epoch_%d_te.pickle' % epoch), "wb") as f:
                pickle.dump(cp_epoch_out, f)
            with open(os.path.join(args.exp, '..', 'pca_epoch_%d_te.pickle' % epoch), "wb") as f:
                pickle.dump(pca_features_te, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)

