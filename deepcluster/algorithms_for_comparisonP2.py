import numpy as np
import torch
import torch.nn as nn
from util import AverageMeter, Logger, UnifLabelSampler
from tools import zip_img_label, flatten_list, rebuild_input_patch, rebuild_pred_patch
import torch.nn.functional as F
import time
import os
import paths
import matplotlib.pyplot as plt
from confusion_matrix import conf_mat, roc_curve_macro, plot_conf, plot_conf_best, plot_macro, plot_macro_best
from batch.combine_functions import CombineFunctions
from batch.label_transform_functions.index_0_1_27_for_comparisonP2 import index_0_1_27_for_comparisonP2
from batch.label_transform_functions.relabel_with_threshold_morph_close_for_comparisonP2 import relabel_with_threshold_morph_close_for_comparisonP2
from batch.label_transform_functions.seabed_checker_for_comparisonP2 import seabed_checker_for_comparisonP2
from batch.data_transform_functions.remove_nan_inf import remove_nan_inf_for_comparisonP2
from batch.data_transform_functions.db_with_limits import db_with_limits_for_comparisonP2


def supervised_train_for_comparisonP2(loader, model, crit, opt_body, opt_category, epoch, device, args):
    #############################################################
    # Supervised learning
    supervised_losses = AverageMeter()
    supervised_output_save = []
    supervised_label_save = []
    for i, (input_tensor, label) in enumerate(loader):
        input_tensor = torch.squeeze(input_tensor)
        label = torch.squeeze(label)
        input_var = torch.autograd.Variable(input_tensor.to(device=device, dtype=torch.double))
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


def test_for_comparisonP2(dataloader, model, crit, device, args):
    if args.verbose:
        print('Test')
    test_losses = AverageMeter()
    model.eval()

    test_output_save = []
    test_out_softmax_save = []
    test_label_save = []
    with torch.no_grad():
        for i, (input_tensor, label) in enumerate(dataloader):
            input_tensor = torch.squeeze(input_tensor)
            label = torch.squeeze(label)
            input_var = torch.autograd.Variable(input_tensor.to(device=device, dtype=torch.double))
            label_var = torch.autograd.Variable(label.to(device))
            output = model(input_var)
            loss = crit(output, label_var.long())
            test_losses.update(loss.item(), input_tensor.size(0))
            pred_mat = F.softmax(output, dim=1)

            output_argmax = torch.argmax(output, axis=1)
            test_out_softmax_save.append(pred_mat.data.cpu().numpy())
            test_output_save.append(output_argmax.data.cpu().numpy())
            test_label_save.append(label.data.cpu().numpy())

            if args.verbose and (i % args.display_count) == 0:
                print('{0} / {1}\t'
                      'TEST_Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(dataloader), loss=test_losses))

    test_out_softmax_flat = flatten_list(test_out_softmax_save)
    output_flat = flatten_list(test_output_save)
    label_flat = flatten_list(test_label_save)
    accu_list = [out == lab for (out, lab) in zip(output_flat, label_flat)]
    test_accuracy = sum(accu_list) / len(accu_list)
    return test_losses.avg, test_accuracy, output_flat, label_flat, test_out_softmax_flat

def test_analysis(predictions, predictions_mat, epoch, args):
    path_to_echograms = paths.path_to_echograms()
    labels_origin = torch.load(os.path.join(path_to_echograms, 'label_TEST_60_after_transformation.pt'))
    if np.shape(predictions_mat) == (60, 256, 256, 3):
        predictions_mat = predictions_mat.transpose(0, 3, 1, 2)
    keep_test_idx = np.where(labels_origin > -1)
    labels_vec = labels_origin[keep_test_idx]
    predictions_vec = predictions[keep_test_idx]
    predictions_mat_sampled = predictions_mat[keep_test_idx[0], :, keep_test_idx[1], keep_test_idx[2]]
    fpr, tpr, roc_auc, roc_auc_macro = roc_curve_macro(labels_vec, predictions_mat_sampled)
    prob_mat, mat, f1_score, kappa = conf_mat(ylabel=labels_vec, ypred=predictions_vec, args=args)
    acc_bg, acc_se, acc_ot = prob_mat.diagonal()
    plot_macro(fpr, tpr, roc_auc, epoch, args)
    plot_conf(epoch, prob_mat, mat, f1_score, kappa, args)
    return fpr, tpr, roc_auc, roc_auc_macro, prob_mat, mat, f1_score, kappa, acc_bg, acc_se, acc_ot

def test_and_plot_2019(test_pred_large_2019, test_label_large_2019, epoch, args, idx=2):
    path_to_echograms = paths.path_to_echograms()
    data_2019, label_2019, patch_loc = torch.load(os.path.join(path_to_echograms, 'data_label_patch_loc_te_2019_%d.pt' % idx))
    data_transform = CombineFunctions([remove_nan_inf_for_comparisonP2, db_with_limits_for_comparisonP2])
    label_transform = CombineFunctions([index_0_1_27_for_comparisonP2, relabel_with_threshold_morph_close_for_comparisonP2, seabed_checker_for_comparisonP2])
    boxsize = 5
    plt.figure(figsize=(boxsize * patch_loc[1], boxsize * patch_loc[0] * 4))
    for i in range(len(data_2019)):
        l = label_2019[i]
        d = data_2019[i]
        d, l = label_transform(d, l)
        d, l = data_transform(d, l)
        dim = np.shape(l)

        labels_rgb = np.ones((dim[0], dim[1], 3))
        seabed = np.where(l == -1)
        sandeel = np.where(l == 1)
        other = np.where(l == 2)
        labels_rgb[sandeel[0], sandeel[1], 0] = 0
        labels_rgb[sandeel[0], sandeel[1], 1] = 0  # sandeel blue
        labels_rgb[other[0], other[1], 1] = 0
        labels_rgb[other[0], other[1], 2] = 0  # other red
        labels_rgb[seabed[0], seabed[1], 0] = 1
        labels_rgb[seabed[0], seabed[1], 1] = 1
        labels_rgb[seabed[0], seabed[1], 2] = 1  # seabed gray 0.7

        pred = test_pred_large_2019[i]
        pred_rgb = np.ones((dim[0], dim[1], 3))
        pred_sandeel = np.where(pred == 1)
        pred_other = np.where(pred == 2)
        pred_rgb[pred_sandeel[0], pred_sandeel[1], 0] = 0
        pred_rgb[pred_sandeel[0], pred_sandeel[1], 1] = 0  # sandeel blue
        pred_rgb[pred_other[0], pred_other[1], 1] = 0
        pred_rgb[pred_other[0], pred_other[1], 2] = 0  # other red
        # pred_rgb[seabed[0], seabed[1], 0] = 1
        # pred_rgb[seabed[0], seabed[1], 1] = 1
        # pred_rgb[seabed[0], seabed[1], 2] = 1  # seabed gray

        lbb = test_label_large_2019[i]
        lbb_rgb = np.ones((dim[0], dim[1], 3))
        lbb_sandeel = np.where(lbb == 1)
        lbb_other = np.where(lbb == 2)
        lbb_rgb[lbb_sandeel[0], lbb_sandeel[1], 0] = 0
        lbb_rgb[lbb_sandeel[0], lbb_sandeel[1], 1] = 0  # sandeel blue
        lbb_rgb[lbb_other[0], lbb_other[1], 1] = 0
        lbb_rgb[lbb_other[0], lbb_other[1], 2] = 0  # other red
        lbb_rgb[seabed[0], seabed[1], 0] = 1
        lbb_rgb[seabed[0], seabed[1], 1] = 1
        lbb_rgb[seabed[0], seabed[1], 2] = 1  # seabed gray


        plt.subplot(patch_loc[0] * 4, patch_loc[1], i + 1)
        plt.imshow(labels_rgb)

        plt.subplot(patch_loc[0] * 4, patch_loc[1], patch_loc[1] * patch_loc[0] + i + 1)
        plt.imshow(pred_rgb)

        plt.subplot(patch_loc[0] * 4, patch_loc[1], patch_loc[1] * patch_loc[0] * 2 + i + 1)
        plt.imshow(lbb_rgb)

        plt.subplot(patch_loc[0] * 4, patch_loc[1], patch_loc[1] * patch_loc[0] * 3 + i + 1)
        plt.imshow(d[-1])

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(os.path.join(args.pred_2019, '%d_data_pred_label_2019_%d_patch.pdf' % (epoch, idx)))
    plt.close()
    return


def compute_features_for_comparisonP2(dataloader, model, N, device, args):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    model.eval()
    # discard the label information in the dataloader
    input_tensors = []
    labels = []
    with torch.no_grad():
         for i, (input_tensor, label) in enumerate(dataloader):
            input_tensor = torch.squeeze(input_tensor)
            label = torch.squeeze(label)
            end = time.time()
            input_var = torch.autograd.Variable(input_tensor.to(device, dtype=torch.double))
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * args.for_comparisonP2_batchsize: (i + 1) * args.for_comparisonP2_batchsize] = aux
            else:
                # special treatment for final batch
                features[i * args.for_comparisonP2_batchsize:] = aux
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


def semi_train_for_comparisonP2(loader, semi_loader, model, fd, crit_pseudo, crit_sup, opt_body, opt_category, epoch, device, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    semi_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, ((input_tensor, label), pseudo_target, imgidx) in enumerate(loader):

        input_var = torch.autograd.Variable(input_tensor.to(device=device, dtype=torch.double))
        pseudo_target_var = torch.autograd.Variable(pseudo_target.to(device,  non_blocking=True))
        output = model(input_var)
        loss = crit_pseudo(output, pseudo_target_var.long())

        # record loss
        losses.update(loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt_body.zero_grad()
        loss.backward()
        opt_body.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % args.display_count) == 0:
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
    model.category_layer.to(device=device, dtype=torch.double)

    category_save = os.path.join(args.exp, 'category_layer.pth.tar')
    if os.path.isfile(category_save):
        category_layer_param = torch.load(category_save)
        model.category_layer.load_state_dict(category_layer_param)

    semi_output_save = []
    semi_label_save = []
    for i, (input_tensor, label) in enumerate(semi_loader):
        input_tensor = torch.squeeze(input_tensor)
        label = torch.squeeze(label)
        input_var = torch.autograd.Variable(input_tensor.to(device=device, dtype=torch.double))
        label_var = torch.autograd.Variable(label.to(device,  non_blocking=True))

        output = model(input_var)
        semi_loss = crit_sup(output, label_var.long())

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
        if args.verbose and (i % args.display_count) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'SEMI_Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(semi_loader), loss=semi_losses))

    semi_output_flat = flatten_list(semi_output_save)
    semi_label_flat = flatten_list(semi_label_save)
    semi_accu_list = [out == lab for (out, lab) in zip(semi_output_flat, semi_label_flat)]
    semi_accuracy = sum(semi_accu_list)/len(semi_accu_list)
    return losses.avg, semi_losses.avg, semi_accuracy
