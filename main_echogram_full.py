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
from batch.data_transform_functions.remove_nan_inf import remove_nan_inf_img
from batch.data_transform_functions.db_with_limits import db_with_limits_img
from batch.combine_functions import CombineFunctions
from classifier_linearSVC import SimpleClassifier
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


# def cluster_acc(Y_pred, Y):
#     assert Y_pred.size == Y.size
#     D = max(Y_pred.max(), Y.max())+1
#     w = np.zeros((D,D), dtype=np.int64)
#     for i in range(Y_pred.size):
#         w[Y_pred[i], Y[i]] += 1
#     ind = linear_sum_assignment(w.max() - w)
#     return sum([w[i,j] for i,j in zip(ind[0], ind[1])])*1.0/Y_pred.size, w

def parse_args():
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16', 'vgg16_tweak'], default='vgg16_tweak',
                        help='CNN architecture (default: vgg16)')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=100,
                        help='number of cluster for k-means (default: 10000)')
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
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--save_epoch', default=30, type=int,
                        help='save features every epoch number (default: 20)')
    parser.add_argument('--batch', default=16, type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--pca', default=32, type=int,
                        help='pca dimension (default: 128)')
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
    parser.add_argument('--sampler_probs', type=list, default=None,
                        help='[bg, sh27, sbsh27, sh01, sbsh01], default=[1, 1, 1, 1, 1]')
    parser.add_argument('--resume',
                        default=os.path.join(current_dir, '..', 'checkpoint.pth.tar'), type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--exp', type=str,
                        default=current_dir, help='path to exp folder')
    parser.add_argument('--optimizer', type=str, metavar='OPTIM',
                        choices=['Adam', 'SGD'], default='Adam', help='optimizer_choice (default: Adam)')

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
    if args.optimizer is 'Adam':
        print('Adam optimizer: top_layer')
        optimizer_tl = torch.optim.Adam(
            model.top_layer.parameters(),
            lr=args.lr_Adam,
            betas=(0.5, 0.99),
            weight_decay=10**args.wd,
        )
    else:
        print('SGD optimizer: top_layer')
        optimizer_tl = torch.optim.SGD(
            model.top_layer.parameters(),
            lr=args.lr_SGD,
            momentum= args.momentum,
            weight_decay=10**args.wd,
        )

    end = time.time()
    input_tensors = []
    labels = []
    pseudo_targets = []
    outputs = []
    imgidxes = []

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
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

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

    # input_tensors = np.concatenate(input_tensors, axis=0)
    # pseudo_targets = np.concatenate(pseudo_targets, axis=0)
    # outputs = np.concatenate(outputs, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # imgidxes = np.concatenate(imgidxes, axis=0)
    # tr_epoch_out = [input_tensors, pseudo_targets, outputs, labels, losses.avg, imgidxes]
    # return losses.avg, tr_epoch_out
    return losses.avg

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
    # idx_2000 = np.random.choice(3000, size=2000, replace=False).tolist()
    # sampler_2000 = []
    # for i in range(5):
    #     sampler_2000.append([samplers_train[i][idx] for idx in idx_2000])
    # torch.save(sampler_2000, 'samplers_2000.pt')
    # path_to_echograms = "/Users/changkyu/Documents/GitHub/echogram/memmap/memmap_set"
    # bg = torch.load(os.path.join(path_to_echograms, 'numpy_bg_2999.pt')) + \
    #      torch.load(os.path.join(path_to_echograms, 'numpy_bg_5999.pt'))
    # bg_idx = np.random.choice(np.arange(len(bg)), size=3000, replace=False)
    # bg = [bg[idx] for idx in bg_idx]
    #
    # sbsh01 = torch.load(os.path.join(path_to_echograms, 'numpy_sbsh01_2999.pt')) +\
    #          torch.load(os.path.join(path_to_echograms, 'numpy_sbsh01_5999.pt')) +\
    #          torch.load(os.path.join(path_to_echograms, 'numpy_sbsh01_8999.pt')) +\
    #          torch.load(os.path.join(path_to_echograms, 'numpy_sbsh01_11999.pt')) +\
    #          torch.load(os.path.join(path_to_echograms, 'numpy_sbsh01_12667.pt'))
    # sbsh01_idx = np.random.choice(np.arange(len(sbsh01)), size=3000, replace=False)
    # sbsh01 = [sbsh01[idx] for idx in sbsh01_idx]
    #
    # sbsh27 = torch.load(os.path.join(path_to_echograms, 'numpy_sbsh27_2999.pt'))+\
    #          torch.load(os.path.join(path_to_echograms, 'numpy_sbsh27_3079.pt'))
    # sbsh27_idx = np.random.choice(np.arange(len(sbsh27)), size=3000, replace=False)
    # sbsh27 = [sbsh27[idx] for idx in sbsh27_idx]
    #
    # sh01 = torch.load(os.path.join(path_to_echograms, 'numpy_sh01_2999.pt'))+\
    #        torch.load(os.path.join(path_to_echograms, 'numpy_sh01_4046.pt'))
    # sh01_idx = np.random.choice(np.arange(len(sh01)), size=3000, replace=False)
    # sh01 = [sh01[idx] for idx in sh01_idx]
    #
    # sh27 = torch.load(os.path.join(path_to_echograms, 'numpy_sh27_2999.pt'))+\
    #        torch.load(os.path.join(path_to_echograms, 'numpy_sh27_3549.pt'))
    # sh27_idx = np.random.choice(np.arange(len(sh27)), size=3000, replace=False)
    # sh27 = [sh27[idx] for idx in sh27_idx]
    # samplers_train = [bg, sh27, sbsh27, sh01, sbsh01]
    # torch.save(samplers_train, 'samplers_3000.pt')
    # samplers_train = [bg, sh27, sbsh27, sh01, sbsh01]
    # def sample_align(samplers):
    #     num_samples = []
    #     new_samplers = []
    #     for i in range(len(samplers)):
    #         num_samples.append(len(samplers[i]))
    #     max_num_sample = np.min(num_samples)
    #     print(max_num_sample)
    #     for i in range(len(samplers)):
    #         new_samplers.append(np.random.choice(samplers[i], size=max_num_sample, replace=False))
    #     return new_samplers
    path_to_echograms = paths.path_to_echograms()
    samplers_train = torch.load(os.path.join(path_to_echograms, 'samplers_1000.pt'))
    augmentation = CombineFunctions([add_noise_img, flip_x_axis_img])
    data_transform = CombineFunctions([remove_nan_inf_img, db_with_limits_img])

    dataset_cp = DatasetImg(
        samplers_train,
        5000,
        args.sampler_probs,
        augmentation_function=augmentation,
        data_transform_function=data_transform)
    return dataset_cp


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
    cluster_log = Logger(os.path.join(args.exp,  '..', 'clusters.pickle'))

    # # Create echogram sampling index
    print('Sample echograms.')
    end = time.time()
    dataset_cp = sampling_echograms_full(args)
    dataloader_cp = torch.utils.data.DataLoader(dataset_cp,
                                                shuffle=False,
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
    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):

        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())) # End with linear(512*128) in original vgg)
                                                                                 # ReLU in .classfier() will follow later
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

        linear_svc = SimpleClassifier(epoch, cp_epoch_out, tr_size=5, iteration=20)
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Classify. accu.: {1:.3f} \n'
                  'Pairwise classify. accu: {2} \n'
                  .format(epoch, linear_svc.whole_score, linear_svc.pair_score))

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
            shuffle=False,
            num_workers=args.workers,
            sampler=sampler_train,
            pin_memory=True,
        )

        # set last fully connected layer
        mlp = list(model.classifier.children()) # classifier that ends with linear(512 * 128)
        mlp.append(nn.ReLU().to(device))
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
        with torch.autograd.set_detect_anomaly(True):
            # loss, tr_epoch_out = train(train_dataloader, model, criterion, optimizer, epoch, device=device, args=args)
            loss = train(train_dataloader, model, criterion, optimizer, epoch, device=device, args=args)
        print('Train time: {0:.2f} s'.format(time.time() - end))

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
                  'ConvNet tr_loss: {2:.3f} \n'
                  'Clustering loss: {3:.3f} \n'
                  .format(epoch, time.time() - end, loss, clustering_loss))

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

        # print('epoch: ', type(epoch), epoch)
        # print('loss: ', type(loss), loss)
        # print('linear_svc.whole_score: ', type(linear_svc.whole_score), linear_svc.whole_score)
        # print('linear_svc.pair_score: ', type(linear_svc.pair_score), linear_svc.pair_score)
        # print('clustering_loss: ', type(clustering_loss), clustering_loss)

        loss_collect[0].append(epoch)
        loss_collect[1].append(loss)
        loss_collect[2].append(linear_svc.whole_score)
        loss_collect[3].append(linear_svc.pair_score)
        loss_collect[4].append(clustering_loss)
        with open(os.path.join(args.exp, '..', 'loss_collect.pickle'), "wb") as f:
            pickle.dump(loss_collect, f)

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)

if __name__ == '__main__':
    args = parse_args()
    main(args)