import numpy as np

from batch.augmentation.flip_x_axis import flip_x_axis
from batch.augmentation.add_noise import add_noise
from data.echogram import get_echograms
from batch.dataset import Dataset
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

def partition_data(echograms, partition='year', portion_train_test=0.8, portion_train_val=0.75):
    # Choose partitioning of data by specifying 'partition' == 'random' OR 'year'

    if partition == 'train_only':
        train, val, test = echograms, [], []

    if partition == 'random':
        # Random partition of all echograms

        # Set random seed to get the same partition every time
        np.random.seed(seed=10)
        np.random.shuffle(echograms)
        train = echograms[:int(portion_train_test * portion_train_val* len(echograms))]
        val = echograms[int(portion_train_test * portion_train_val * len(echograms)):int(portion_train_test * len(echograms))]
        test = echograms[int(portion_train_test * len(echograms)):]

        # Reset random seed to generate random crops during training
        np.random.seed(seed=None)

    elif partition == 'year':
        train = list(filter(lambda x: any([year in x.name for year in
                                           ['D2014','D2015']]), echograms))
        val = list(filter(lambda x: any([year in x.name for year in
                                         ['D2016', 'D2017']]), echograms))
        test = list(filter(lambda x: any([year in x.name for year in
                                          ['D2013', 'D2018']]), echograms))
    else:
        print("Parameter 'partition' must equal 'random' or 'year'")

    print('Train:', len(train), ' Val:', len(val), ' Test:', len(test))

    return train, val, test

# def gen_patches(sampler, window_size, frequencies, num_samples,
#                 sampler_probs,
#                 augmentation,
#                 label_transform,
#                 data_transform):
#     if (len(sampler) > 1):
#         dataset = Dataset(sampler,
#                           window_size,
#                           frequencies,
#                           num_samples,
#                           sampler_probs,
#                           augmentation_function=augmentation,
#                           label_transform_function=label_transform,
#                           data_transform_function=data_transform)
#     else:
#         dataset = DatasetSingleSampler(sampler,
#                                         window_size,
#                                         frequencies,
#                                         num_samples,
#                                         augmentation_function=augmentation,
#                                         label_transform_function=label_transform,
#                                         data_transform_function=data_transform)
#
#     images = []
#     labels = []
#     center_locations = []
#     echogram_names = []
#     for i in range(num_samples):
#         data, label, center_location, echogram_name = dataset[i]
#         images.append(data)
#         labels.append(label)
#         center_locations.append(center_location)
#         echogram_names.append(echogram_name)
#     return images, labels, center_locations, echogram_names
#
#
# def gen_patch_figure(idx, images, labels, center_locations, echogram_names):
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.imshow(labels[idx])
#     plt.colorbar()
#     plt.title('%s_%s' %(center_locations[idx], echogram_names[idx]))
#     plt.savefig('%d.jpg' % (idx))
#
#     for i in range(4):
#         plt.figure()
#         plt.imshow(images[idx][i])
#         plt.colorbar()
#         plt.title('%s_%s' % (center_locations[idx], echogram_names[idx]))
#         plt.savefig('%d_%d.jpg' %(idx, frequencies[i]))


####################################################################################


# window_dim = 32
# frequencies = [18, 38, 120, 200]
# window_size = [window_dim, window_dim]
# partition = 'year'
# num_samples = 100
#
# # Load echograms and create partition for train/test/val
# echograms = get_echograms(frequencies=frequencies, minimum_shape=window_dim)
# echograms_train, echograms_val, echograms_test = partition_data(echograms, partition, portion_train_test=0.8, portion_train_val=0.75)
#
# # num_workers = 10
# # print('num_workers: ', num_workers)
#
#
# sampler_bg_train = Background(echograms_train, window_size)
# sampler_sb_train = Seabed(echograms_train, window_size)
# sampler_sh27_train = Shool(echograms_train, 27)
# sampler_sh01_train = Shool(echograms_train, 1)
# sampler_sbsh27_train = ShoolSeabed(echograms_train, window_dim // 2, 27)
# sampler_sbsh01_train = ShoolSeabed(echograms_train, window_dim // 2, 1)
#
# sampler_bg_val = Background(echograms_val, window_size)
# sampler_sb_val = Seabed(echograms_val, window_size)
# sampler_sh27_val = Shool(echograms_val, 27)
# sampler_sh01_val = Shool(echograms_val, 1)
# sampler_sbsh27_val = ShoolSeabed(echograms_val, window_dim // 2, 27)
# sampler_sbsh01_val = ShoolSeabed(echograms_val, window_dim // 2, 1)
#
#
# augmentation = CombineFunctions([add_noise, flip_x_axis])
# label_transform = CombineFunctions([index_0_1_27, relabel_with_threshold_morph_close])
# data_transform = CombineFunctions([remove_nan_inf, db_with_limits])
#
#
# gen_patch_args = {'window_size': window_size,
#                    'frequencies': frequencies,
#                   'num_samples': num_samples,
#                    'augmentation': augmentation,
#                    'label_transform':label_transform,
#                    'data_transform':data_transform}
#
# images, labels, center_locations, echogram_names = \
#     gen_patches(sampler_in, **gen_patch_args)

# non_zero = []
# for i in range(num_samples):
#     if not np.allclose(labels[i], 0):
#         non_zero.append(i)
# print(len(non_zero))
#
# for idx in non_zero:
#     gen_patch_figure(idx, images, labels, center_locations, echogram_names)
#
#
# indices = idx = np.random.choice(100, 10)
# for idx in indices:
#     gen_patch_figure(idx, images, labels, center_locations, echogram_names)

# see full images

# non_zero_idx = non_zero[0]
# idx = 14
# full_echo_search(idx, full_images, full_labels, echogram_names)
# e = search_echogram(echograms_train, echogram_names[idx])
# e.visualize(predictions=None,
#                   labels_original=None,
#                   labels_refined=None,
#                   labels_korona=None,
#                   pred_contrast=1.0,
#                   frequencies=None,
#                   draw_seabed=True,
#                   show_labels=True,
#                   show_object_labels=True,
#                   show_grid=True,
#                   show_name=True,
#                   show_freqs=True,
#                   show_labels_str=True,
#                   show_predictions_str=True,
#                   return_fig=False,
#                   figure=None)
#