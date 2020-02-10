import numpy as np
import copy
from utils.np import getGrid, linear_interpolation, nearest_interpolation

class DatasetSingleSampler():

    def __init__(self, sampler, window_size, frequencies,
                 n_samples = 1000,
                 # sampler_probs=None,
                 augmentation_function=None,
                 label_transform_function=None,
                 data_transform_function=None):
        """
        A dataset is used to draw random samples
        :param sampler: The sampler used to draw samples
        :param window_size: expected window size
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        :param augmentation_function:
        :param label_transform_function:
        :param data_transform_function:
        """

        self.sampler = sampler
        self.window_size = window_size
        self.n_samples = n_samples
        self.frequencies = frequencies
        self.augmentation_function = augmentation_function
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function

    def __getitem__(self, index):

        #Draw coordinate and echogram with sampler
        center_location, echogram = self.sampler.get_sample()

        #Get data/labels-patches
        data, labels = get_crop(echogram, center_location, self.window_size, self.frequencies)

        # Apply augmentation
        if self.augmentation_function is not None:
            data, labels, echogram = self.augmentation_function(data, labels, echogram)

        # Apply label-transform-function
        if self.label_transform_function is not None:
            data, labels, echogram = self.label_transform_function(data, labels, echogram)

        # Apply data-transform-function
        if self.data_transform_function is not None:
            data, labels, echogram, frequencies = self.data_transform_function(data, labels, echogram, self.frequencies)

        labels = labels.astype('int16')
        # return data, labels, center_location, echogram.name
        return data, labels

    def __len__(self):
        return self.n_samples

def get_crop(echogram, center_location, window_size, freqs):
    """
    Returns a crop of data around the pixels specified in the center_location.

    """
    # Get grid sampled around center_location
    grid = getGrid(window_size) + np.expand_dims(np.expand_dims(center_location, 1), 1)

    channels = []
    for f in freqs:

        # Interpolate data onto grid
        memmap = echogram.data_memmaps(f)[0]
        data = linear_interpolation(memmap, grid, boundary_val=0, out_shape=window_size)
        del memmap

        # Set non-finite values (nan, positive inf, negative inf) to zero
        if np.any(np.invert(np.isfinite(data))):
            data[np.invert(np.isfinite(data))] = 0

        channels.append(np.expand_dims(data, 0))
    channels = np.concatenate(channels, 0)

    labels = nearest_interpolation(echogram.label_memmap(), grid, boundary_val=-100, out_shape=window_size)

    return channels, labels

def get_full(echogram, freqs):
    channels = []
    shape = echogram.shape
    grid = getGrid(shape)
    for f in freqs:
        # Interpolate data onto grid
        memmap = echogram.data_memmaps(f)[0]
        data = linear_interpolation(memmap, grid, boundary_val=0, out_shape=shape)
        # Set non-finite values (nan, positive inf, negative inf) to zero
        if np.any(np.invert(np.isfinite(data))):
            data[np.invert(np.isfinite(data))] = 0

        channels.append(np.expand_dims(data, 0))
    channels = np.concatenate(channels, 0)

    labels = nearest_interpolation(echogram.label_memmap(), grid, boundary_val=-100, out_shape=shape)
    return channels, labels

