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
    parser.add_argument('--nmb_category', type=int, default=3,
                        help='number of ground truth classes(category)')
    parser.add_argument('--pretrain_epoch', default=100, type=int,
                        help='epoch count for pretrain_with few labels (semi-sup)')
    parser.add_argument('--lr_Adam', default=1e-4, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--lr_Adam_pretrain', default=1e-4, type=float,
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
    parser.add_argument('--batch', default=32, type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--pca', default=32, type=int,
                        help='pca dimension (default: 128)')
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
    parser.add_argument('--semi_ratio', type=float, default=0.1, help='ratio of the labeled samples')

    # parser.add_argument('--iteration_train', type=int, default=1200,
    #                     help='num_tr_iterations per one batch and epoch')
    # parser.add_argument("--mode", default='client')
    # parser.add_argument("--port", default=52162)

    return parser.parse_args(args=[])

def zip_img_label(img_tensors, labels):
    img_label_pair = []
    for i, zips in enumerate(zip(img_tensors, labels)):
        img_label_pair.append(zips)
    print('num_pairs: ', len(img_label_pair))
    return img_label_pair

def pretrain(semi_loader, model, crit, opt, epoch, device, args):
    batch_time = AverageMeter()
    semi_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    #############################################################
    # Semi-supervised learning
    for i, (input_tensor, label) in enumerate(semi_loader):
        input_var = torch.autograd.Variable(input_tensor.to(device))
        label_var = torch.autograd.Variable(label.to(device,  non_blocking=True))

        output = model(input_var)
        semi_loss = crit(output, label_var.long())

        # record loss
        semi_losses.update(semi_loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        semi_loss.backward()
        opt.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 3) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'PreTrain: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(semi_loader), loss=semi_losses))

    return semi_losses.avg


def train(loader, semi_loader, model, fd, crit, opt, epoch, device, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    semi_losses = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    #############################################################
    # Semi-supervised learning
    model.cluster_layer = None
    model.category_layer = nn.Sequential(
        nn.Linear(fd, args.nmb_category),
        nn.Softmax(dim=1),
    )
    model.category_layer.load_state_dict(torch.load('./category_layer_pretrain.pt'))
    model.category_layer.to(device)

    for i, (input_tensor, label) in enumerate(semi_loader):
        input_var = torch.autograd.Variable(input_tensor.to(device))
        label_var = torch.autograd.Variable(label.to(device,  non_blocking=True))

        output = model(input_var)
        semi_loss = crit(output, label_var.long())

        # record loss
        semi_losses.update(semi_loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        semi_loss.backward()
        opt.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 3) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'SEMI_Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(semi_loader), loss=semi_losses))

    ##############################

    end = time.time()
    input_tensors = []
    labels = []
    pseudo_targets = []
    outputs = []
    imgidxes = []

    model.category_layer = None
    model.cluster_layer = nn.Sequential(
        nn.Linear(fd, args.nmb_cluster),  # nn.Linear(4096, num_cluster),
        nn.Softmax(dim=1),  # should be removed and replaced by ReLU for category_layer
    )
    # load_state_dict ?
    model.cluster_layer[0].weight.data.normal_(0, 0.01)
    model.cluster_layer[0].bias.data.zero_()
    model.cluster_layer = model.cluster_layer.double()
    model.cluster_layer.to(device)

    for i, ((input_tensor, label), pseudo_target, imgidx) in enumerate(loader):
        data_time.update(time.time() - end)

        # save checkpoint
        n = len(loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.exp, '..',
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)

        input_var = torch.autograd.Variable(input_tensor.to(device))
        pseudo_target_var = torch.autograd.Variable(pseudo_target.to(device,  non_blocking=True))

        output = model(input_var)
        loss = crit(output, pseudo_target_var.long())

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
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'PSEUDO_Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time, loss=losses))

        input_tensors.append(input_tensor.data.cpu().numpy())
        pseudo_targets.append(pseudo_target.data.cpu().numpy())
        outputs.append(output.data.cpu().numpy())
        labels.append(label)
        imgidxes.append(imgidx)

        input_tensors = []
        labels = []
        pseudo_targets = []
        outputs = []
        imgidxes = []

    return losses.avg, semi_losses.avg


def compute_features(dataloader, model, N, device, args):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    input_tensors = []
    labels = []
    with torch.no_grad():
         for i, (input_tensor, label) in enumerate(dataloader):
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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose and (i % 10) == 0:
                print('{0} / {1}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      .format(i, len(dataloader), batch_time=batch_time))

            input_tensors.append(input_tensor.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())
         input_tensors = np.concatenate(input_tensors, axis=0)
         labels = np.concatenate(labels, axis=0)
         return features, input_tensors, labels


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


def test(dataloader, model, fd, crit, device, args):
    if args.verbose:
        print('Test')
    batch_time = AverageMeter()
    test_losses = AverageMeter()
    end = time.time()

    model.cluster_layer = None
    model.category_layer = nn.Sequential(
        nn.Linear(fd, args.nmb_category),  # nn.Linear(4096, num_cluster),
        nn.Softmax(dim=1),  # should be removed and replaced by ReLU for category_layer
    )
    # load_state_dict ?
    model.category_layer[0].weight.data.normal_(0, 0.01)
    model.category_layer[0].bias.data.zero_()
    model.category_layer = model.category_layer.double()
    model.category_layer.to(device)

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
# def sampling_echograms_eval(args):
#     # echograms_eval = get_echograms(years=[2019], frequencies=[18, 38, 120, 200],
#     #                                minimum_shape=int(args.window_dim * 5), maximum_shape=int(args.window_dim * 100))
#     # stride_eval = [args.stride, args.stride]
#     # gap_eval = GetAllPatches(echograms_eval, window_size, stride_eval, fish_type=[1, 27], random_offset_ratio=1024, phase='eval')
#     # echograms_eval = gap_eval.target_echograms
#     window_size = [args.window_dim, args.window_dim]
#     path_to_eval = paths.path_to_eval()
#     eval_dir_names = ['2019847-D20190512-T140210', '2019847-D20190512-T161218', '2019847-D20190512-T143430', '2019847-D20190512-T153731']
#     echograms_eval = [Echogram(os.path.join(path_to_eval, e)) for e in eval_dir_names]
#     sampler_eval = SampleFull(echograms_eval, window_size, args.stride)
#     data_transform = CombineFunctions([remove_nan_inf_img, db_with_limits_img])
#     dataset_eval = DatasetGrid(
#         sampler_eval,
#         window_size,
#         args.frequencies,
#         data_transform_function=data_transform)
#     return dataset_eval

# def evaluate(loader, model, device, args):
#     print("####################### Start evaluate #######################")
#     fd = int(model.top_layer[0].weight.size()[1])
#     torch.save(model.top_layer.state_dict(), './top_layer_eval.pt')
#     N = loader.dataset.__len__()
#     input_tensors = []
#
#     features = []
#     outputs = []
#
#     with torch.no_grad():
#         for i, (input_tensor, _) in enumerate(loader):
#             input_tensor.double()
#             input_var = torch.autograd.Variable(input_tensor.to(device))
#
#             # get vectorial expression
#             model.top_layer = None
#             aux = model(input_var)
#
#             # recall top layer
#             model.top_layer = nn.Sequential(
#                 nn.Linear(fd, args.nmb_cluster),
#                 nn.Softmax(dim=1),
#             )
#             model.top_layer.load_state_dict(torch.load('./top_layer_eval.pt'))
#             model.top_layer = model.top_layer.double()
#             model.top_layer.to(device)
#
#             out = model.top_layer_forward(aux)
#
#             input_tensors.extend(input_tensor.data.cpu().numpy())
#             features.extend(aux.data.cpu().numpy())
#             outputs.extend(out.data.cpu().numpy())
#
#             # if i == 0:
#             #     features = np.zeros((N, aux.shape[1]), dtype='float32')
#             #     outputs = np.zeros((N, out.shape[1]), dtype='float32')
#             # aux = aux.astype('float32')
#             # out = out.astype('float32')
#             # if i < len(loader) - 1:
#             #     features[i * args.batch: (i + 1) * args.batch] = aux
#             #     outputs[i * args.batch: (i + 1) * args.batch] = out
#             # else:
#             #     features[i * args.batch:] = aux
#             #     outputs[i * args.batch:] = out
#
#     echograms = loader.dataset.sampler_test.echograms
#     center_locations = loader.dataset.sampler_test.center_locations
#     eval_epoch_out = [features, outputs, input_tensors, echograms, center_locations]
#     print("####################### End evaluate #######################")
#     return eval_epoch_out

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
    fd = int(model.cluster_layer[0].weight.size()[1])  # due to transpose, fd is input dim of W (in dim, out dim)
    model.category_layer = None
    model.cluster_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model = model.double()
    model.to(device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

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
    cluster_log = Logger(os.path.join(args.exp,  '..', 'clusters.pickle'))

    # # Create echogram sampling index
    print('Sample echograms.')
    end = time.time()
    dataset_cp, dataset_semi = sampling_echograms_full(args)
    dataloader_cp = torch.utils.data.DataLoader(dataset_cp,
                                                shuffle=False,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                drop_last=False,
                                                pin_memory=True)

    dataloader_semi = torch.utils.data.DataLoader(dataset_semi,
                                                shuffle=True,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                drop_last=False,
                                                pin_memory=True)

    dataset_test = sampling_echograms_test(args)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                shuffle=True,
                                                batch_size=args.batch,
                                                num_workers=args.workers,
                                                drop_last=False,
                                                pin_memory=True)

    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster, args.pca)
    #                   deepcluster = clustering.Kmeans(no.cluster, dim.pca)

    loss_collect = [[], [], [], [], []]
    nmi_save = []

    # for evaluation
    # dataset_eval = sampling_echograms_eval(args)
    # eval_dataloader = torch.utils.data.DataLoader(dataset_eval,
    #                                               batch_size=args.batch,
    #                                               shuffle=False,
    #                                               num_workers=args.workers,
    #                                               pin_memory=True,
    #                                               )

    ######### pretrain

    # pretrain
    if args.start_epoch == 0:
        print("Start pretrain upto %d epochs" % args.start_epoch)
        model.cluster_layer = None
        model.category_layer = nn.Sequential(
            nn.Linear(fd, args.nmb_category),
            nn.Softmax(dim=1),
        )
        model.category_layer[0].weight.data.normal_(0, 0.01)
        model.category_layer[0].bias.data.zero_()
        model.category_layer = model.category_layer.double()
        model.category_layer.to(device)
        optimizer_pretrain = torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr_Adam_pretrain,
            betas=(0.5, 0.99),
            weight_decay=10 ** args.wd,
        )
        pretrain_loss_save = []
        for epoch in range(args.pretrain_epoch):
            pretrain_loss = pretrain(dataloader_semi, model, criterion, optimizer_pretrain, epoch, device, args)
            pretrain_loss_save.append(pretrain_loss)
        torch.save(model.category_layer.state_dict(), './category_layer_pretrain.pt')
        with open(os.path.join(args.exp, '..', 'pretrain_loss.pickle'), "wb") as f:
            pickle.dump(pretrain_loss_save, f)

    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        # remove head
        model.cluster_layer = None
        model.category_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1]) # remove ReLU at classifier [:-1]
        # get the features for the whole dataset
        features_train, input_tensors_train, labels_train = compute_features(dataloader_cp, model, len(dataset_cp), device=device, args=args)

        # cluster the features
        print('Cluster the features')
        end = time.time()
        clustering_loss, pca_features = deepcluster.cluster(features_train, verbose=args.verbose)
        # deepcluster.cluster(features_train, verbose=args.verbose)
        print('Cluster time: {0:.2f} s'.format(time.time() - end))

        nan_location = np.isnan(pca_features)
        inf_location = np.isinf(pca_features)
        if (not np.allclose(nan_location, 0)) or (not np.allclose(inf_location, 0)):
            print('PCA: Feature NaN or Inf found. Nan count: ', np.sum(nan_location), ' Inf count: ', np.sum(inf_location))
            print('Skip epoch ', epoch)
            torch.save(pca_features, 'pca_NaN_%d.pth.tar' % epoch)
            torch.save(features_train, 'feature_NaN_%d.pth.tar' % epoch)
            continue

        # save patches per epochs
        cp_epoch_out = [features_train, deepcluster.images_lists, deepcluster.images_dist_lists, input_tensors_train,
                        labels_train]

        # linear_svc = SimpleClassifier(epoch, cp_epoch_out, tr_size=5, iteration=20)
        # if args.verbose:
        #     print('###### Epoch [{0}] ###### \n'
        #           'Classify. accu.: {1:.3f} \n'
        #           'Pairwise classify. accu: {2} \n'
        #           .format(epoch, linear_svc.whole_score, linear_svc.pair_score))

        if (epoch % args.save_epoch == 0):
            end = time.time()
            with open(os.path.join(args.exp, '..', 'cp_epoch_%d.pickle' % epoch), "wb") as f:
                pickle.dump(cp_epoch_out, f)
            with open(os.path.join(args.exp, '..', 'pca_epoch_%d.pickle' % epoch), "wb") as f:
                pickle.dump(pca_features, f)
            print('Feature save time: {0:.2f} s'.format(time.time() - end))

        # assign pseudo-labels
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
            shuffle=True,
            num_workers=args.workers,
            sampler=sampler_train,
            pin_memory=True,
        )

        # Recover classifier with ReLU (that is not used in clustering)
        mlp = list(model.classifier.children()) # classifier that ends with linear(512 * 128). No ReLU at the end
        mlp.append(nn.ReLU(inplace=True).to(device))
        model.classifier = nn.Sequential(*mlp)

        # train network with clusters as pseudo-labels
        end = time.time()
        with torch.autograd.set_detect_anomaly(True):
            pseudo_loss, semi_loss = train(train_dataloader, dataloader_semi, model, fd, criterion, optimizer, epoch, device=device, args=args)
        print('Train time: {0:.2f} s'.format(time.time() - end))
        end = time.time()
        test_loss = test(dataloader_test, model, fd, criterion, device, args)

        # if (epoch % args.save_epoch == 0):
        #     end = time.time()
        #     with open(os.path.join(args.exp, '..', 'tr_epoch_%d.pickle' % epoch), "wb") as f:
        #         pickle.dump(tr_epoch_out, f)
        #     print('Save train time: {0:.2f} s'.format(time.time() - end))

        # Accuracy with training set (output vs. pseudo label)
        # accuracy_tr = np.mean(tr_epoch_out[1] == np.argmax(tr_epoch_out[2], axis=1))

        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Pseudo tr_loss: {2:.3f} \n'
                  'SEMI tr_loss: {3:.3f} \n'
                  'TEST loss: {4:.3f} \n'
                  'Clustering loss: {5:.3f} \n'.format(epoch, time.time() - end, pseudo_loss, semi_loss, test_loss, clustering_loss))

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
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(args.exp,  '..', 'checkpoint.pth.tar'))

        # evaluation: echogram reconstruction
        # if (epoch % args.save_epoch == 0):
        #     eval_epoch_out = evaluate(eval_dataloader, model, device=device, args=args)
        #     with open(os.path.join(args.exp, '..', 'eval_epoch_%d.pickle' % epoch), "wb") as f:
        #         pickle.dump(eval_epoch_out, f)

        # print('epoch: ', type(epoch), epoch)
        # print('loss: ', type(loss), loss)
        # print('Test loss: ', type(test_loss), test_loss)
        # print('linear_svc.whole_score: ', type(linear_svc.whole_score), linear_svc.whole_score)
        # print('linear_svc.pair_score: ', type(linear_svc.pair_score), linear_svc.pair_score)
        # print('clustering_loss: ', type(clustering_loss), clustering_loss)

        loss_collect[0].append(epoch)
        loss_collect[1].append(pseudo_loss)
        loss_collect[2].append(semi_loss)
        loss_collect[3].append(clustering_loss)
        loss_collect[4].append(test_loss)
        # loss_collect[2].append(linear_svc.whole_score)
        # loss_collect[3].append(linear_svc.pair_score)

        with open(os.path.join(args.exp, '..', 'loss_collect.pickle'), "wb") as f:
            pickle.dump(loss_collect, f)

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)

if __name__ == '__main__':
    args = parse_args()
    main(args)

'''
data_transform = CombineFunctions([remove_nan_inf_img, db_with_limits_img])

for j in range(10):
    dbdata = db(samplers_semi[-1][j])
    print('######## ', j)
    for i in range(4):
        print(dbdata[i].min(), dbdata[i].max())

data_f27 = data_transform(samplers_semi[-1][0])
'''
