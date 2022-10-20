import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.getcwd())
sys.path.append('./deepcluster')
from deepcluster.samplers_for_comparisonP2 import sampling_echograms_2019_for_comparisonP2_pixel

imgidx = [1, 5, 6, 9]
section_idx = [[12, 13, 14, 15, 16], [29, 30, 31, 32, 33], [21, 22, 23, 24, 25, 26, 27, 28, 29], [14, 15, 16]]
semi_ratios = [20, 25, 30, 35, 40, 100]

for semi_ratio in semi_ratios:
    get_dir = '/Users/changkyu/Desktop/Paper2/com_semi_%dp' % semi_ratio
    for k, (img, section) in enumerate(zip(imgidx, section_idx)):
        pred_2019_pixel = torch.load(os.path.join(get_dir, '%d_pred_2019_pixel_%d.tar' % (semi_ratio, img)))
        dataset_2019_pixel, label_2019, patch_loc = sampling_echograms_2019_for_comparisonP2_pixel(echogram_idx=img, get_section=section)
        full_size = dataset_2019_pixel.full_size
        print('semi ratio: ', semi_ratio, '\t img idx: ', img, '\t section: ', section, '\t\t original size', full_size)
        recon_pixel = np.zeros((3, full_size[0], full_size[1]))
        for i, cls in enumerate(pred_2019_pixel):
            col = i % (full_size[1] - 32 + 1)
            row = i // (full_size[1] - 32 + 1)
            recon_pixel[cls, row:row+32, col:col+32] += 1
        recon_pixel_argmax = np.argmax(recon_pixel, axis=0)

        visual_recon = np.ones((full_size[0], full_size[1], 3))
        visual_recon[recon_pixel_argmax == 1] = [0, 0, 1]
        visual_recon[recon_pixel_argmax == 2] = [1, 0, 0]
        n_patches = len(section)
        split_recon = np.hsplit(visual_recon, n_patches)

        plt.close()
        fig, ax = plt.subplots(1, n_patches, sharex=True, sharey=True, figsize=(n_patches*3, 3))
        for i in range(n_patches):
            ax[i].imshow(split_recon[i])
            ax[i].set_xticks([], [])
            ax[i].set_yticks([], [])

        plt.tight_layout()
        plt.savefig('%d_%d.pdf' % (semi_ratio, img))