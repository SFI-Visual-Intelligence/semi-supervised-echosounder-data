# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# from sklearn.metrics.cluster import normalized_mutual_info_score
# import time
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from scipy.optimize import linear_sum_assignment
# import matplotlib.pyplot as plt
# import copy
# import faiss


current_dir = os.getcwd()

# if current_dir[-1] is not 'p':
#     os.chdir(os.path.join(current_dir, 'semi', '10p'))
#     current_dir = os.getcwd()

sys.path.append(os.path.join(current_dir, '..', '..', 'deepcluster'))

import clustering
import models
from tools import zip_img_label, flatten_list, rebuild_input_patch, rebuild_pred_patch
from util import AverageMeter, Logger, UnifLabelSampler
from algorithms_for_comparisonP2 import supervised_train_for_comparisonP2, test_for_comparisonP2, compute_features_for_comparisonP2, semi_train_for_comparisonP2, test_analysis, test_and_plot_2019, test_for_comparisonP2_pixel
from samplers_for_comparisonP2 import sampling_echograms_full_for_comparisonP2, sampling_echograms_test_for_comparisonP2, sampling_echograms_2019_for_comparisonP2, sampling_echograms_2019_for_comparisonP2_pixel

# from confusion_matrix import conf_mat, roc_curve_macro, plot_conf, plot_conf_best, plot_macro, plot_macro_best
# import paths
# from batch.data_transform_functions.remove_nan_inf import remove_nan_inf_for_comparisonP2
# from batch.data_transform_functions.db_with_limits import db_with_limits_for_comparisonP2
# from batch.combine_functions import CombineFunctions
# from batch.label_transform_functions.index_0_1_27_for_comparisonP2 import index_0_1_27_for_comparisonP2
# from batch.label_transform_functions.relabel_with_threshold_morph_close_for_comparisonP2 import relabel_with_threshold_morph_close_for_comparisonP2
# from batch.label_transform_functions.seabed_checker_for_comparisonP2 import seabed_checker_for_comparisonP2
# from batch.dataset import DatasetImg_for_comparisonP2
# from batch.dataset import DatasetGrid
# from batch.samplers.sampler_test import SampleFull
# from batch.samplers.get_all_patches import GetAllPatches
# from data.echogram import Echogram
# from clustering import preprocess_features
# from batch.augmentation.flip_x_axis import flip_x_axis_img
# from batch.augmentation.add_noise import add_noise_img
# from batch.dataset import DatasetImg
# from batch.dataset import DatasetImgUnbal
# from classifier_linearSVC import SimpleClassifier


def parse_args():
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    parser.add_argument('--start_epoch', default=100, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--for_comparisonP2_batchsize', default=32, type=int, help='minibatch_size for comparison P2')
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
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--pretrain_epoch', type=int, default=0,
                        help='number of pretrain epochs to run (default: 200)')
    parser.add_argument('--save_epoch', default=50, type=int,
                        help='save features every epoch number (default: 20)')
    parser.add_argument('--batch', default=1, type=int,
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
    parser.add_argument('--semi_ratio', type=float, default=0.30, help='ratio of the labeled samples')
    parser.add_argument('--f1_avg', type=str, default='weighted', help='the way computing f1-score')
    return parser.parse_args(args=[])


def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)
    criterion_pseudo = nn.CrossEntropyLoss()
    criterion_sup = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([10, 300, 250]).to(device=device, dtype=torch.double))

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
    # '''
    # ########################################
    # ########################################
    # Create echogram sampling index
    # ########################################
    # ########################################'''
    #
    # print('Sample echograms.')
    # dataset_cp, dataset_semi = sampling_echograms_full_for_comparisonP2(args) # For comparison (paper #2)
    #
    # dataloader_cp = torch.utils.data.DataLoader(dataset_cp,
    #                                             shuffle=False,
    #                                             batch_size=args.batch,
    #                                             num_workers=args.workers,
    #                                             drop_last=False,
    #                                             pin_memory=True)
    #
    # dataloader_semi = torch.utils.data.DataLoader(dataset_semi,
    #                                             shuffle=False,
    #                                             batch_size=args.batch,
    #                                             num_workers=args.workers,
    #                                             drop_last=False,
    #                                             pin_memory=True)
    #
    #
    # dataset_te = sampling_echograms_test_for_comparisonP2()
    #
    # dataloader_test = torch.utils.data.DataLoader(dataset_te,
    #                                             shuffle=False,
    #                                             batch_size=args.batch,
    #                                             num_workers=args.workers,
    #                                             drop_last=False,
    #                                             pin_memory=True)
    #
    #
    #
    # deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster, args.pca)
    #
    # resume_path = os.path.join(args.exp, '..', '%dp' % int(args.semi_ratio * 100), 'checkpoints')
    # # resume_path = '/Users/changkyu/Desktop/Springfield_backup/deepcluster_P2/semi/feature_maker_20p/checkpoints'
    # args.resume = os.path.join(resume_path, '%d_checkpoint.pth.tar' % args.start_epoch)
    # resume_path = '/storage/Users/changkyu/Desktop/Springfield_backup/deepcluster_P2/semi/feature_maker_20p/checkpoints'

    resume_path = '/storage/p1_results_for_p2/semi%d' % int(100 * args.semi_ratio)
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
            # category_save = os.path.join(resume_path, '%d_category_layer.pth.tar' % args.start_epoch)
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
    #
    # # creating checkpoint repo
    # exp_check = os.path.join(args.exp, 'checkpoints')
    # if not os.path.isdir(exp_check):
    #     os.makedirs(exp_check)
    #
    # exp_test = os.path.join(args.exp, 'test')
    # for dir_2 in ['2019', 'pred']:
    #     dir_to_make = os.path.join(exp_test, dir_2)
    #     if not os.path.isdir(dir_to_make):
    #         os.makedirs(dir_to_make)
    #
    # '''
    # #######################
    # #######################
    # MAIN TRAINING
    # #######################
    # #######################'''
    # for epoch in range(args.start_epoch, args.epochs):
    #     print('#####################  Start training at Epoch %d ################'% epoch)
    #     model.classifier = nn.Sequential(*list(model.classifier.children())[:-1]) # remove ReLU at classifier [:-1]
    #     model.cluster_layer = None
    #     model.category_layer = None
    #
    #     '''
    #     #######################
    #     #######################
    #     PSEUDO-LABEL GENERATION
    #     #######################
    #     #######################
    #     '''
    #     print('Cluster the features')
    #     features_train, input_tensors_train, labels_train = compute_features_for_comparisonP2(dataloader_cp, model, len(dataset_cp) * args.for_comparisonP2_batchsize, device=device, args=args)
    #     clustering_loss, pca_features = deepcluster.cluster(features_train, verbose=args.verbose)
    #
    #     nan_location = np.isnan(pca_features)
    #     inf_location = np.isinf(pca_features)
    #     if (not np.allclose(nan_location, 0)) or (not np.allclose(inf_location, 0)):
    #         print('PCA: Feature NaN or Inf found. Nan count: ', np.sum(nan_location), ' Inf count: ', np.sum(inf_location))
    #         print('Skip epoch ', epoch)
    #         torch.save(pca_features, 'tr_pca_NaN_%d.pth.tar' % epoch)
    #         torch.save(features_train, 'tr_feature_NaN_%d.pth.tar' % epoch)
    #         continue
    #
    #     print('Assign pseudo labels')
    #     size_cluster = np.zeros(len(deepcluster.images_lists))
    #     for i,  _list in enumerate(deepcluster.images_lists):
    #         size_cluster[i] = len(_list)
    #     print('size in clusters: ', size_cluster)
    #     img_label_pair_train = zip_img_label(input_tensors_train, labels_train)
    #     train_dataset = clustering.cluster_assign(deepcluster.images_lists,
    #                                               img_label_pair_train)  # Reassigned pseudolabel
    #
    #     # uniformly sample per target
    #     sampler_train = UnifLabelSampler(int(len(train_dataset)),
    #                                deepcluster.images_lists)
    #
    #     train_dataloader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.for_comparisonP2_batchsize, #args.batch
    #         shuffle=False,
    #         num_workers=args.workers,
    #         sampler=sampler_train,
    #         pin_memory=True,
    #     )
    #
    #     '''
    #     ####################################################################
    #     ####################################################################
    #     TRSNSFORM MODEL FOR SELF-SUPERVISION // SEMI-SUPERVISION
    #     ####################################################################
    #     ####################################################################
    #     '''
    #     # Recover classifier with ReLU (that is not used in clustering)
    #     mlp = list(model.classifier.children()) # classifier that ends with linear(512 * 128). No ReLU at the end
    #     mlp.append(nn.ReLU(inplace=True).to(device))
    #     model.classifier = nn.Sequential(*mlp)
    #     model.classifier.to(device=device, dtype=torch.double)
    #
    #     '''SELF-SUPERVISION (PSEUDO-LABELS)'''
    #     model.category_layer = None
    #     model.cluster_layer = nn.Sequential(
    #         nn.Linear(fd, args.nmb_cluster),  # nn.Linear(4096, num_cluster),
    #         nn.Softmax(dim=1),  # should be removed and replaced by ReLU for category_layer
    #     )
    #     model.cluster_layer[0].weight.data.normal_(0, 0.01)
    #     model.cluster_layer[0].bias.data.zero_()
    #     # model.cluster_layer = model.cluster_layer.double()
    #     model.cluster_layer.to(device=device, dtype=torch.double)
    #
    #     ''' train network with clusters as pseudo-labels '''
    #     with torch.autograd.set_detect_anomaly(True):
    #         pseudo_loss, semi_loss, semi_accuracy = semi_train_for_comparisonP2(train_dataloader,
    #                                                                             dataloader_semi,
    #                                                                             model, fd,
    #                                                                             criterion_pseudo,
    #                                                                             criterion_sup,
    #                                                                             optimizer_body, optimizer_category,
    #                                                                             epoch, device=device, args=args)
    #     # save checkpoint
    #     if epoch % args.checkpoints == 0:
    #         path = os.path.join(
    #             args.exp,
    #             'checkpoints',
    #              str(epoch) + '_checkpoint.pth.tar',
    #         )
    #         if args.verbose:
    #             print('Save checkpoint at: {0}'.format(path))
    #         torch.save({'epoch': epoch,
    #                     'arch': args.arch,
    #                     'state_dict': model.state_dict(),
    #                     'optimizer_body': optimizer_body.state_dict(),
    #                     'optimizer_category': optimizer_category.state_dict(),
    #                     }, path)
    #         torch.save(model.category_layer.state_dict(), os.path.join(args.exp, 'checkpoints', '%d_category_layer.pth.tar'% epoch))
    #
    #     # save running checkpoint
    #     torch.save({'epoch': epoch,
    #                 'arch': args.arch,
    #                 'state_dict': model.state_dict(),
    #                 'optimizer_body': optimizer_body.state_dict(),
    #                 'optimizer_category': optimizer_category.state_dict(),
    #                 },
    #                os.path.join(args.exp, 'checkpoint.pth.tar'))
    #     torch.save(model.category_layer.state_dict(), os.path.join(args.exp, 'category_layer.pth.tar'))
    #
    #     '''
    #     ##############
    #     ##############
    #     # TEST phase
    #     ##############
    #     ##############
    #     '''
    #     test_loss, test_accuracy, test_pred, test_label, test_pred_softmax = test_for_comparisonP2(dataloader_test, model, criterion_sup, device, args)
    #     test_pred_large = rebuild_pred_patch(test_pred)
    #     test_softmax_large = rebuild_pred_patch(test_pred_softmax)
    #     test_label_large = rebuild_pred_patch(test_label)
    #
    #     '''Save prediction of the test set'''
    #     if (epoch % args.save_epoch == 0):
    #         with open(os.path.join(args.exp, 'test', 'pred', 'pred_softmax_label_epoch_%d_te.pickle' % epoch), "wb") as f:
    #             pickle.dump([test_pred_large, test_softmax_large, test_label_large], f)
    #
    #     fpr, \
    #     tpr, \
    #     roc_auc, \
    #     roc_auc_macro, \
    #     prob_mat, \
    #     mat, \
    #     f1_score, \
    #     kappa, \
    #     bg_accu, \
    #     se_accu, \
    #     ot_accu = test_analysis(test_pred_large, test_softmax_large, epoch, args)
    #
    #     if os.path.isfile(os.path.join(args.exp, 'records_te_epoch_patch.pth.tar')):
    #         records_te_epoch = torch.load(os.path.join(args.exp, 'records_te_epoch_patch.pth.tar'))
    #     else:
    #         records_te_epoch = {'epoch': [],
    #                             'fpr': [],
    #                             'tpr': [],
    #                             'roc_auc': [],
    #                             'roc_auc_macro': [],
    #                             'prob_mat': [],
    #                             'mat': [],
    #                             'f1_score': [],
    #                             'kappa': [],
    #                             'BG_accu_epoch': [],
    #                             'SE_accu_epoch': [],
    #                             'OT_accu_epoch': [],
    #                             }
    #     records_te_epoch['epoch'].append(epoch)
    #     records_te_epoch['fpr'].append(fpr)
    #     records_te_epoch['tpr'].append(tpr)
    #     records_te_epoch['roc_auc'].append(roc_auc)
    #     records_te_epoch['roc_auc_macro'].append(roc_auc_macro)
    #     records_te_epoch['prob_mat'].append(prob_mat)
    #     records_te_epoch['mat'].append(mat)
    #     records_te_epoch['f1_score'].append(f1_score)
    #     records_te_epoch['kappa'].append(kappa)
    #     records_te_epoch['BG_accu_epoch'].append(bg_accu)
    #     records_te_epoch['SE_accu_epoch'].append(se_accu)
    #     records_te_epoch['OT_accu_epoch'].append(ot_accu)
    #     torch.save(records_te_epoch, os.path.join(args.exp, 'records_te_epoch_patch.pth.tar'))
    #
    #     '''
    #     ##############
    #     ##############
    #     # 2019 phase
    #     ##############
    #     ##############
    #     '''
    #
    #     for i in [1, 5, 6, 9]: # needs only 4 samples out of 11
    #         dataset_2019, label_2019, patch_loc = sampling_echograms_2019_for_comparisonP2(echogram_idx=i)
    #
    #         dataloader_2019 = torch.utils.data.DataLoader(dataset_2019,
    #                                                       batch_size=1,
    #                                                       shuffle=False,
    #                                                       num_workers=args.workers,
    #                                                       worker_init_fn=np.random.seed,
    #                                                       drop_last=False,
    #                                                       pin_memory=True)
    #
    #         test_loss_2019, test_accuracy_2019, test_pred_2019, test_label_2019, test_pred_softmax_2019 = test_for_comparisonP2(dataloader_2019, model, criterion_sup, device, args)
    #         test_pred_large_2019 = rebuild_pred_patch(test_pred_2019)
    #         test_softmax_large_2019 = rebuild_pred_patch(test_pred_softmax_2019)
    #         test_label_large_2019 = rebuild_pred_patch(test_label_2019)
    #
    #         test_and_plot_2019(test_pred_large_2019, test_label_large_2019, epoch, args, idx=i)

    '''
    ##############
    ##############
    # 2019 pixel
    ##############
    ##############
    '''
    imgidx =  [1, 5, 6, 9]
    section_idx = [[12, 13, 14, 15 ,16], [29, 30, 31, 32, 33], [21, 22, 23, 24, 25, 26, 27, 28, 29], [14, 15, 16]]

    for (img, section) in zip(imgidx, section_idx): # needs only 4 samples out of 11
        dataset_2019_pixel, label_2019, patch_loc = sampling_echograms_2019_for_comparisonP2_pixel(echogram_idx=img, get_section=section)
        print('2019 img_idx: %d' % img)
        dataloader_2019_pixel = torch.utils.data.DataLoader(dataset_2019_pixel,
                                                      batch_size=32,
                                                      shuffle=False,
                                                      num_workers=args.workers,
                                                      worker_init_fn=np.random.seed,
                                                      drop_last=False,
                                                      pin_memory=True)

        test_pred_2019_pixel = test_for_comparisonP2_pixel(dataloader_2019_pixel, model, device, args)
        torch.save(test_pred_2019_pixel, '%d_pred_2019_pixel_%d.tar' % (int(args.semi_ratio * 100), img))

        # test_and_plot_2019(test_pred_large_2019, test_label_large_2019, epoch, args, idx=i)


if __name__ == '__main__':
    args = parse_args()
    main(args)


