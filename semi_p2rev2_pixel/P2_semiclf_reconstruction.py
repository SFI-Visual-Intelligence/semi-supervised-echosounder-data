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
sys.path.append(os.path.join(current_dir, 'deepcluster'))


from algorithms_for_comparisonP2 import test_for_comparisonP2_pixel
from samplers_for_comparisonP2 import sampling_echograms_test_for_comparisonP2_pixel

def parse_args():
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    parser.add_argument('--workers', default=2, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch', default=900, type=int,
                        help='mini-batch size (default: 16)')
    return parser.parse_args(args=[])

args = parse_args()

dataset_te = sampling_echograms_test_for_comparisonP2_pixel(stride=1)

dataloader_test = torch.utils.data.DataLoader(dataset_te,
                                            shuffle=False,
                                            batch_size=args.batch,
                                            num_workers=args.workers,
                                            drop_last=False,
                                            pin_memory=True)


def reconstruction(input, stride=1, maxsize=256, kernel=32, num_te_patch=60):
    canvas = np.zeros((num_te_patch, 4, maxsize, maxsize))
    max_idx = maxsize-kernel

    for index, c in enumerate(input):
        patch_idx = index // (((max_idx // stride) + 1) ** 2)
        row_and_col = index % (((max_idx // stride) + 1) ** 2)
        row_idx = row_and_col // ((max_idx // stride) + 1)
        col_idx = row_and_col % ((max_idx // stride) + 1)
        if c == -1:
            canvas[patch_idx][3, row_idx * stride: row_idx * stride + kernel, col_idx * stride: col_idx * stride + kernel] += 1
        else:
            canvas[patch_idx][c, row_idx * stride: row_idx * stride + kernel, col_idx * stride: col_idx * stride + kernel] += 1
    return np.argmax(canvas, axis=1)



# for i, (d, l) in enumerate(dataloader_test):
#     if i ==0:
#         ls = l
#     else:
#         ls = np.concatenate((ls, l))
# cc = reconstruction(ls)
# cct = np.argmax(cc, axis=1)
#
# i = 2
# plt.close()
# plt.figure(i)
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(dataset_te.label[i])
# ax[1].imshow(cct[i])
# plt.show()


def reconstruction_visual(canvas):

# if c == 0:
#     continue
# if c == 1:
#     canvas[patch_idx][1:, row_idx*stride: row_idx * stride + kernel, col_idx*stride: col_idx * stride + kernel] = 0
# elif c == 2:
#     canvas[patch_idx][:2, row_idx*stride: row_idx * stride + kernel, col_idx*stride: col_idx * stride + kernel] = 0


