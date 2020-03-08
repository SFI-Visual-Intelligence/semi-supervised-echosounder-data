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

import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler

from batch.augmentation.flip_x_axis import flip_x_axis
from batch.augmentation.add_noise import add_noise
from data.echogram import get_echograms
from batch.dataset import Dataset
from batch.dataset import DatasetVal
from batch.dataset_sampler import DatasetSingleSampler
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

# def fig_patches(dataset_train):
#     import matplotlib.pyplot as plt
#     imgs, label, coord, ecname, labelmap = dataset_train[0]
#     plt.figure(figsize=(30, 30))
#     fig, ((ax1, _1, ax2), (_2, ax5, _3), (ax3, _4, ax4)) = plt.subplots(3, 3)
#     fig.suptitle('label: %d, coord: %s, ename: %s' % (label, coord, ecname))
#     ax1.imshow(imgs[0])
#     ax2.imshow(imgs[1])
#     ax3.imshow(imgs[2])
#     ax4.imshow(imgs[3])
#     ax5.imshow(labelmap)
#     plt.savefig('./test_pics/label_%d_coord_%s_ename_%s.jpg' % (label, coord, ecname))

# for i in range(500):
#     fig_patches(dataset_train)
def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i,j] for i,j in zip(ind[0], ind[1])])*1.0/Y_pred.size, w

def parse_args():
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
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--save_epoch', default=20, type=int,
                        help='save features every epoch number (default: 20)')
    parser.add_argument('--batch', default=32, type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--pca', default=32, type=int,
                        help='pca dimension (default: 16)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='/content/gdrive/My Drive/UiT_phd/deepcluster/deepcluster', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    # parser.add_argument('--resume', default='/storage/deepcluster/checkpoint.pth.tar', type=str, metavar='PATH',
    #                     help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=200,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='/content/gdrive/My Drive/UiT_phd/deepcluster/deepcluster', help='path to exp folder')
    # parser.add_argument('--exp', type=str, default='/storage/deepcluster', help='path to exp folder')
    parser.add_argument('--verbose', type=bool, default=True, help='chatty')
    parser.add_argument('--frequencies', type=list, default=[18, 38, 120, 200],
                        help='4 frequencies [18, 38, 120, 200]')
    parser.add_argument('--window_dim', type=int, default=128,
                        help='window size')
    parser.add_argument('--partition', type=str, default='year',
                        help='echogram partition (tr/val/te) by year')
    parser.add_argument('--iteration_train', type=int, default=2,
                        help='num_tr_iterations per one batch and epoch')
    parser.add_argument('--iteration_val', type=int, default=2,
                        help='num_val_iterations per one batch and  epoch')
    parser.add_argument('--sampler_probs', type=list, default=None,
                        help='[bg, sh27, sbsh27, sh01, sbsh01], default=[1, 1, 1, 1, 1]')
    # parser.add_argument('--iteration_test', type=int, default=100,
    #                     help='num_te_iterations per epoch')
    return parser.parse_args(args=[])

def zip_img_label(img_tensors, labels):
    img_label_pair = []
    for i, zips in enumerate(zip(img_tensors, labels)):
        img_label_pair.append(zips)
    print('num_pairs: ', len(img_label_pair))
    return img_label_pair

def train(loader, model, crit, opt, epoch, device, args):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10**args.wd,
    )

    end = time.time()
    input_tensors = []
    labels = []
    pseudo_targets = []
    outputs = []

    for i, ((input_tensor, label), pseudo_target) in enumerate(loader):
        data_time.update(time.time() - end)

        # save checkpoint
        n = len(loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.exp,
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
        print('loss :', loss)
        print('input_tensor.size(0) :', input_tensor.size(0))
        losses.update(loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 10) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'PSEUDO_Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time, loss=losses))


        input_tensors.append(input_tensor.data.cpu().numpy())
        pseudo_targets.append(pseudo_target.data.cpu().numpy())
        outputs.append(output.data.cpu().numpy())
        labels.append(label)

    input_tensors = np.concatenate(input_tensors, axis=0)
    pseudo_targets = np.concatenate(pseudo_targets, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    labels = np.concatenate(labels, axis=0)
    tr_epoch_out = [input_tensors, pseudo_targets, outputs, labels]
    return losses.avg, tr_epoch_out
    # return losses.avg

def validation(loader, model, crit, epoch, device, args):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    val_losses = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.eval()
    end = time.time()
    input_tensors = []
    labels = []
    outputs = []
    with torch.no_grad():
        for i, (input_tensor, label) in enumerate(loader):
            data_time.update(time.time() - end)
            input_var = torch.autograd.Variable(input_tensor.to(device))
            # target_var = torch.autograd.Variable(target.to(device,  non_blocking=True))

            output = model(input_var)
            # val_loss = crit(output, target_var.long())

            # record loss
            # val_losses.update(val_loss.item(), input_tensor.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if args.verbose and (i % 10) == 0:
            #     print('Epoch: [{0}][{1}/{2}]\t\t'
            #           'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Val_Loss: {val_loss.val:.4f} ({val_loss.avg:.4f})'
            #           .format(epoch, i, len(loader), batch_time=batch_time, val_loss=val_losses))

            input_tensors.append(input_tensor.data.cpu().numpy())
            outputs.append(output.data.cpu().numpy())
            labels.append(label)

        input_tensors = np.concatenate(input_tensors, axis=0)
        labels = np.concatenate(labels, axis=0)
        outputs = np.concatenate(outputs, axis=0)
        val_epoch_out = [input_tensors, labels, outputs]
        return val_losses.avg, val_epoch_out
        # return val_epoch_out

def compute_features(dataloader, model, N, device, args):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    input_tensors = []
    labels = []
    # center_location_heights = []
    # center_location_widths = []
    # ecnames = []
    # labelmaps = []
    with torch.no_grad():
         # for i, (input_tensor, label, center_location, ecname, labelmap) in enumerate(dataloader):
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
            # center_location_heights.append(center_location[0].data.cpu().numpy())
            # center_location_widths.append(center_location[1].data.cpu().numpy())
            # ecnames.append(ecname)
            # labelmaps.append(labelmap.data.cpu().numpy())

         input_tensors = np.concatenate(input_tensors, axis=0)
         labels = np.concatenate(labels, axis=0)
         # center_location_heights = np.concatenate(center_location_heights, axis=0)
         # center_location_widths = np.concatenate(center_location_widths, axis=0)
         # ecnames = np.concatenate(ecnames, axis=0)
         # labelmaps = np.concatenate(labelmaps, axis=0)
         # return features, labels, (center_location_heights, center_location_widths), ecnames, input_tensors, labelmaps
         return features, input_tensors, labels

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

    model = models.__dict__[args.arch](sobel=False, bn=True, out=args.nmb_cluster)
    fd = int(model.top_layer[0].weight.size()[1])  # due to transpose, fd is input dim of W (in dim, out dim)
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model = model.double()
    model.to(device)
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
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
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(args.exp, 'clusters.pickle'))

    # load dataset
    end = time.time()
    window_size = [args.window_dim, args.window_dim]
    echograms = get_echograms(frequencies=args.frequencies, minimum_shape=args.window_dim)
    echograms_train, echograms_val, echograms_test = cps.partition_data(echograms, args.partition, portion_train_test=0.8, portion_train_val=0.75)

    sampler_bg_train = Background(echograms_train, window_size)
    # sampler_sb_train = Seabed(echograms_train, window_size)
    sampler_sh27_train = Shool(echograms_train, window_size, 27)
    sampler_sbsh27_train = ShoolSeabed(echograms_train, window_size, args.window_dim//4, fish_type=27)
    sampler_sh01_train = Shool(echograms_train, window_size, 1)
    sampler_sbsh01_train = ShoolSeabed(echograms_train, window_size, args.window_dim//4, fish_type=1)

    sampler_bg_val = Background(echograms_val, window_size)
    # sampler_sb_val = Seabed(echograms_val, window_size)
    sampler_sh27_val = Shool(echograms_val, window_size, 27)
    sampler_sbsh27_val = ShoolSeabed(echograms_val, window_size, args.window_dim//4, fish_type=27)
    sampler_sh01_val = Shool(echograms_val, window_size, 1)
    sampler_sbsh01_val = ShoolSeabed(echograms_val, window_size, args.window_dim//4, fish_type=1)

    samplers_train = [sampler_bg_train, #sampler_sb_train,
                      sampler_sh27_train, sampler_sbsh27_train,
                      sampler_sh01_train, sampler_sbsh01_train]

    samplers_val = [sampler_bg_val, # sampler_sb_val,
                    sampler_sh27_val, sampler_sbsh27_val,
                    sampler_sh01_val, sampler_sbsh01_val]

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

    dataset_val = DatasetVal(
        samplers_val,
        window_size,
        args.frequencies,
        args.batch * args.iteration_val,
        args.sampler_probs,
        augmentation_function=None,
        label_transform_function=label_transform,
        data_transform_function=data_transform)

    val_dataloader = torch.utils.data.DataLoader(dataset_val,
                                             shuffle=True,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)


    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader_cp = torch.utils.data.DataLoader(dataset_train,
                                             shuffle=False,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)
    # next(iter(dataloader_cp)) => input_tensors, label, center_locations, ecnames, labelmaps

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster, args.pca)
    #                   deepcluster = clustering.Kmeans(no.cluster, dim.pca)

    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())) # End with linear(512*128) in original vgg)
                                                                                 # ReLU in .classfier() will follow later
        # get the features for the whole dataset
        # features_train, labels_train, center_locations_train, ecnames_train, input_tensors_train, labelmaps_train \
        features_train, input_tensors_train, labels_train = compute_features(dataloader_cp, model, len(dataset_train), device=device, args=args)

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features_train, verbose=args.verbose)

        # deepcluster.images_dist_lists
        # deepcluster.images_lists

        # save patches per epochs
        if ((epoch+1) % args.save_epoch == 0):
            # cp_epoch_out = [features_train, deepcluster.images_lists, deepcluster.images_dist_lists, input_tensors_train, labels_train, center_locations_train, ecnames_train, labelmaps_train]
            cp_epoch_out = [features_train, deepcluster.images_lists, deepcluster.images_dist_lists, input_tensors_train, labels_train]
            with open("./cp_epoch_%d.pickle" % epoch, "wb") as f:
                pickle.dump(cp_epoch_out, f)

        # if epoch % args.save_epoch == 0:
        #     with open("./dist_lists_%d.pickle" % epoch, "wb") as g:
        #         pickle.dump(deepcluster.images_dist_lists, g)


        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')

        size_cluster = np.zeros(len(deepcluster.images_lists))
        for i,  _list in enumerate(deepcluster.images_lists):
            size_cluster[i] = len(_list)
        print('size in clusters: ', size_cluster)

        img_label_pair_train = zip_img_label(input_tensors_train, labels_train)
        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                  img_label_pair_train)  # Reassigned pseudolabel
                                                                        # e.g. (7, 7, 3, 3, 3, 2, 2) -> (0, 0, 1, 1, 1, 2, 2)_

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

        # set last fully connected layer
        mlp = list(model.classifier.children()) # classifier that ends with linear(512 * 128)
        mlp.append(nn.ReLU(inplace=True).to(device))
        model.classifier = nn.Sequential(*mlp)

        model.top_layer = nn.Sequential(
            nn.Linear(fd, args.nmb_cluster),
            nn.Softmax(dim=1),
            )
        # model.top_layer = nn.Linear(fd, args.nmb_cluster)
        model.top_layer[0].weight.data.normal_(0, 0.01)
        model.top_layer[0].bias.data.zero_()
        model.top_layer = model.top_layer.double()
        model.top_layer.to(device)

        # train network with clusters as pseudo-labels

        end = time.time()
        loss, tr_epoch_out = train(train_dataloader, model, criterion, optimizer, epoch, device=device, args=args)


        ###############################################################
        # print('Extract validation samples')
        # # input_tensors_val = []
        # # labels_val = []
        # # for i in range(args.batch * args.iteration_val):
        # #     input_tensor_val, label_val = dataset_val[i]
        # #     input_tensors_val.append(input_tensor_val)
        # #     labels_val.append(label_val)
        # #
        # # val_images_lists = [[] for i in range(args.nmb_cluster)]
        # # for idx, label in enumerate(labels_val):
        # #     val_images_lists[label].append(idx)
        # #
        # # val_dataset = clustering.cluster_assign(val_images_lists,
        # #                                           input_tensors_val)
        # # # uniformly sample per target
        # # sampler_val = UnifLabelSampler(int(len(val_dataset)),
        # #                                  val_images_lists)
        # #
        # #
        # #
        # # val_dataloader = torch.utils.data.DataLoader(
        # #     val_dataset,
        # #     batch_size=args.batch,
        # #     shuffle=False,
        # #     num_workers=args.workers,
        # #     sampler=sampler_val,
        # #     pin_memory=True,
        # # )
        #
        # val_loss, val_epoch_out = validation(val_dataloader, model, criterion, epoch, device=device, args=args)
        ###############################################################

        if ((epoch+1) % args.save_epoch == 0):
            with open("./tr_epoch_%d.pickle" % epoch, "wb") as f:
                pickle.dump(tr_epoch_out, f)
            # with open("./val_epoch_%d.pickle" % epoch, "wb") as g:
            #     pickle.dump(val_epoch_out, g)

        # Accuracy with training set (output vs. pseudo label)
        accuracy_tr = np.mean(tr_epoch_out[1] == np.argmax(tr_epoch_out[2], axis=1))
        # Accuracy with validaton set (output vs. true label + Hunagarian)
        # accuracy_val, w = cluster_acc(np.argmax(val_epoch_out[2], axis=1), val_epoch_out[1])

        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {1:.3f} \n'
                  'ConvNet tr_loss: {2:.3f} \n'
                  'ConvNet tr_acc: {3:.3f} \n'
                  .format(epoch, time.time() - end, clustering_loss, loss, accuracy_tr))

            # print('###### Epoch [{0}] ###### \n'
            #       'Time: {1:.3f} s\n'
            #       'Clustering loss: {2:.3f} \n'
            #       'ConvNet tr_loss: {3:.3f} \n'
            #       'ConvNet val_loss: {4:.3f} \n'
            #       'ConvNet tr_acc: {5:.3f} \n'
            #       'ConvNet val_acc: {6:.3f} \n'
            #       .format(epoch, time.time() - end, clustering_loss, loss,val_loss, accuracy_tr, accuracy_val))

            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(args.exp, 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)

if __name__ == '__main__':
    args = parse_args()
    main(args)