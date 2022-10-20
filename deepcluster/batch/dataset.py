import numpy as np

from utils.np import getGrid, linear_interpolation, nearest_interpolation

# class Dataset_nouse():
#
#     def __init__(self, samplers, window_size, frequencies,
#                  n_samples = 1000,
#                  sampler_probs=None,
#                  augmentation_function=None,
#                  label_transform_function=None,
#                  data_transform_function=None):
#         """
#         A dataset is used to draw random samples
#         :param samplers: The samplers used to draw samples
#         :param window_size: expected window size
#         :param n_samples:
#         :param frequencies:
#         :param sampler_probs:
#         :param augmentation_function:
#         :param label_transform_function:
#         :param data_transform_function:
#         """
#
#         self.samplers = samplers
#         self.window_size = window_size
#         self.n_samples = n_samples
#         self.frequencies = frequencies
#         self.sampler_probs = sampler_probs
#         self.augmentation_function = augmentation_function
#         self.label_transform_function = label_transform_function
#         self.data_transform_function = data_transform_function
#
#         # Normalize sampling probabillities
#         if self.sampler_probs is None:
#             self.sampler_probs = np.ones(len(samplers))
#         self.sampler_probs = np.array(self.sampler_probs)
#         self.sampler_probs = np.cumsum(self.sampler_probs).astype(float)
#         self.sampler_probs /= np.max(self.sampler_probs)
#
#     def __getitem__(self, index):
#         #Select which sampler to use
#         i = np.random.rand()
#         sample_idx = np.where(i < self.sampler_probs)[0][0]
#         sampler = self.samplers[sample_idx]
#
#         #Draw coordinate and echogram with sampler
#         center_location, echogram = sampler.get_sample()
#
#         # Adjust coordinate by random shift in y and x direction
#         # center_location[0] += np.random.randint(-self.window_size[0]//2, self.window_size[0]//2 + 1)
#         # center_location[1] += np.random.randint(-self.window_size[1]//2, self.window_size[1]//2 + 1)
#         # center_location[0] += np.random.randint(-self.window_size[0]//4, self.window_size[0]//4 + 1)
#         # center_location[1] += np.random.randint(-self.window_size[1]//4, self.window_size[1]//4 + 1)
#
#         #Get data/labels-patches
#         data, labels = get_crop(echogram, center_location, self.window_size, self.frequencies)
#
#         # Apply augmentation
#         if self.augmentation_function is not None:
#             data, labels, echogram = self.augmentation_function(data, labels, echogram)
#
#         # Apply label-transform-function
#         if self.label_transform_function is not None:
#             data, labels, echogram = self.label_transform_function(data, labels, echogram)
#
#         # Apply data-transform-function
#         if self.data_transform_function is not None:
#             data, labels, echogram, frequencies = self.data_transform_function(data, labels, echogram, self.frequencies)
#
#         labels = labels.astype('int16')
#         return data, sample_idx, center_location, echogram.name, labels
#         # return data, labels, center_location, echogram.name
#         # return data, labels
#
#     def __len__(self):
#         return self.n_samples

def patch_splitter(sample, after_size=32, before_size=256, step=32, half_idx=0):
    slice_indices = np.arange(after_size, before_size, step=step)
    patches = np.asarray(np.split(sample, slice_indices, axis=-1))
    if len(np.shape(patches)) == 4: # echosounder patch
        patches = np.reshape(np.asarray(np.split(patches, slice_indices, axis=-2)), (-1, 4, 32, 32))
    elif len(np.shape(patches)) == 3: # label patch
        patches = np.reshape(np.asarray(np.split(patches, slice_indices, axis=-2)), (-1, 32, 32))

    if half_idx == 0:
        patches = patches [:32]
    else:
        patches = patches[32:]
    return patches

def patch_splitter_pixel(index, sample, out_size, in_size):
    assert (len(np.shape(sample)) == 3) or (len(np.shape(sample)) == 2), "check input sample (echosounder data or segmentation label)"
    depth = index // (in_size[1] - out_size + 1)
    width = index % (in_size[1] - out_size + 1)

    if len(sample.shape) == 3:  # if input is the echosounder data with 4 channel (4, 256, 256)
        return sample[:, depth:depth+out_size, width:width+out_size]
    else:
        return sample[depth:depth+out_size, width:width+out_size]

def label_scalar_single(l_patch, criteria=16):
    unique = np.unique(l_patch)
    if np.isin(-1, unique):
        l_scalar = -1
    else:
        l_vec = np.bincount(l_patch.astype(int).reshape(-1), minlength=3)
        if (l_vec[1] > criteria) or (l_vec[2] > criteria):
            if l_vec[1] > l_vec[2]:
                l_scalar = 1
            else:
                l_scalar = 2
        else:
            l_scalar = 0
    return np.asarray(l_scalar)

def label_scalar(l_patches, criteria=16):
    l_scalars = []
    for l_patch in l_patches:
        unique = np.unique(l_patch)
        if np.isin(-1, unique):
            l_scalar = -1
        else:
            l_vec = np.bincount(l_patch.astype(int).reshape(-1), minlength=3)
            if (l_vec[1] > criteria) or (l_vec[2] > criteria):
                if l_vec[1] > l_vec[2]:
                    l_scalar = 1
                else:
                    l_scalar = 2
            else:
                l_scalar = 0
        l_scalars.append(l_scalar)

    return np.asarray(l_scalars)

class Dataset():

    def __init__(self, samplers, window_size, frequencies,
                 n_samples = 1000,
                 sampler_probs=None,
                 augmentation_function=None,
                 label_transform_function=None,
                 data_transform_function=None):
        """
        A dataset is used to draw random samples
        :param samplers: The samplers used to draw samples
        :param window_size: expected window size
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        :param augmentation_function:
        :param label_transform_function:
        :param data_transform_function:
        """

        self.samplers = samplers
        self.window_size = window_size
        self.n_samples = n_samples
        self.frequencies = frequencies
        self.sampler_probs = sampler_probs
        self.augmentation_function = augmentation_function
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function

        # Normalize sampling probabillities
        if self.sampler_probs is None:
            self.sampler_probs = np.ones(len(samplers))
        self.sampler_probs = np.array(self.sampler_probs)
        self.sampler_probs = np.cumsum(self.sampler_probs).astype(float)
        self.sampler_probs /= np.max(self.sampler_probs)

    def __getitem__(self, index):
        #Select which sampler to use
        i = np.random.rand()
        sample_idx = np.where(i < self.sampler_probs)[0][0]
        sampler = self.samplers[sample_idx]

        #Draw coordinate and echogram with sampler
        center_location, echogram = sampler.get_sample()

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
        return data, sample_idx

    def __len__(self):
        return self.n_samples


class DatasetImgUnbal():
    def __init__(self, samplers,
                 sampler_probs=None,
                 augmentation_function=None,
                 data_transform_function=None):

        self.samplers = samplers
        self.n_samples = int(len(self.samplers) * len(self.samplers[0]))
        self.sampler_probs = sampler_probs
        self.augmentation_function = augmentation_function
        self.data_transform_function = data_transform_function

    def __getitem__(self, index):
        #Select which sampler to use
        sample_idx = index % len(self.samplers)
        img_idx = index // len(self.samplers)
        data, label = self.samplers[sample_idx][img_idx]
        # Apply augmentation
        if self.augmentation_function is not None:
            data = self.augmentation_function(data)
        # Apply data-transform-function
        if self.data_transform_function is not None:
            data = self.data_transform_function(data)
        return data, label

    def __len__(self):
        return self.n_samples


class DatasetImg():
    def __init__(self, samplers,
                 sampler_probs=None,
                 augmentation_function=None,
                 data_transform_function=None):
        """
        A dataset is used to draw random samples
        :param samplers: The samplers used to draw samples
        :param window_size: expected window size
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        :param augmentation_function:
        :param label_transform_function:
        :param data_transform_function:
        """
        self.samplers = samplers
        self.n_samples = int(len(self.samplers) * len(self.samplers[0]))
        self.sampler_probs = sampler_probs
        self.augmentation_function = augmentation_function
        self.data_transform_function = data_transform_function


    def __getitem__(self, index):
        #Select which sampler to use
        sample_idx = index % len(self.samplers)
        img_idx = index // len(self.samplers)
        data = self.samplers[sample_idx][img_idx]
        # Apply augmentation
        if self.augmentation_function is not None:
            data = self.augmentation_function(data)
        # Apply data-transform-function
        if self.data_transform_function is not None:
            data = self.data_transform_function(data)
        return data, sample_idx

    def __len__(self):
        return self.n_samples


class DatasetImg_for_comparisonP2():
    def __init__(self, data, label,
                 label_transform_function=None,
                 data_transform_function=None):
        """
        A dataset is used to draw random samples
        :param samplers: The samplers used to draw samples
        :param window_size: expected window size
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        :param augmentation_function:
        :param label_transform_function:
        :param data_transform_function:
        """
        self.data = data
        self.label = label
        self.n_samples = len(data) * 2
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function

    def __getitem__(self, index):
        img_idx = index // 2
        half_idx = index % 2   # Only to fit into the batch-size. no scientific meaning behind this.

        data_sample = self.data[img_idx]
        label_sample = self.label[img_idx]

        # Apply data-transform-function
        if self.label_transform_function is not None:
            data_sample, label_sample = self.label_transform_function(data_sample, label_sample)
        # Apply label_augmentation
        if self.data_transform_function is not None:
            data_sample, label_sample = self.data_transform_function(data_sample, label_sample)

        d_patches = patch_splitter(data_sample, half_idx=half_idx)
        l_patches = patch_splitter(label_sample, half_idx=half_idx)
        l_scalars = label_scalar(l_patches)

        return d_patches, l_scalars


    def __len__(self):
        return self.n_samples


class DatasetImg_for_comparisonP2_pixel_test():
    def __init__(self, data,
                 label,
                 stride,
                 label_transform_function,
                 data_transform_function):
        self.data_transform_function = data_transform_function
        self.label_transform_function = label_transform_function
        self.stride = stride
        self.kernel = 32
        self.max_idx = 224
        self.num_256_patches = len(label)
        self.data = []
        self.label = []
        for i in range(len(data)):
            data_filt, label_filt =  self.label_transform_function(data[i], label[i])
            data_filt, label_filt = self.data_transform_function(data_filt, label_filt)
            self.data.append(data_filt)
            self.label.append(label_filt)

    def __getitem__(self, index):
        patch_idx = index // (((self.max_idx // self.stride) +1)**2)
        row_and_col = index % (((self.max_idx // self.stride) +1) **2)
        row_idx = row_and_col // ((self.max_idx // self.stride) +1)
        col_idx = row_and_col % ((self.max_idx // self.stride) +1)
        data_sample = self.data[patch_idx][:, row_idx*self.stride: row_idx*self.stride + self.kernel, col_idx*self.stride: col_idx*self.stride+self.kernel]
        label_sample = self.label[patch_idx][row_idx*self.stride: row_idx*self.stride + self.kernel, col_idx*self.stride: col_idx*self.stride+self.kernel]

        l_scalar = label_scalar_single(label_sample)
        return data_sample, l_scalar

    def __len__(self):
        return self.num_256_patches * ((self.max_idx // self.stride) +1) **2

def patch_cal(label_patch, stride, kernel_size=32):
    len = np.shape(label_patch)[0]
    patch_wind = np.shape(label_patch)[2]
    return len * ((patch_wind - kernel_size)//stride + 1) **2

def index_check(index, data_patch, label_patch, stride, kernel_size=32):
    total_lenn = patch_cal(label_patch, stride, kernel_size)
    max_idx = 224
    kernel = 32
    patch_idx = index // (((max_idx // stride) + 1) ** 2)
    row_and_col = index % (((max_idx // stride) + 1) ** 2)
    row_idx = row_and_col // ((max_idx // stride) + 1)
    col_idx = row_and_col % ((max_idx // stride) + 1)
    print(total_lenn, patch_idx, row_idx, col_idx)
    data_sample = data_patch[patch_idx][:][row_idx * stride: row_idx * stride + kernel, col_idx * stride: col_idx * stride + kernel]
    label_sample = label_patch[patch_idx][row_idx * stride: row_idx * stride + kernel, col_idx * stride: col_idx * stride + kernel]
    return data_sample, label_sample



class DatasetImg_for_comparisonP2_pixel_2019():
    def __init__(self, data,
                 label,
                 get_section,
                 label_transform_function,
                 data_transform_function):
        self.data = data
        self.label = label
        self.get_section = get_section
        self.data_transform_function = data_transform_function
        self.label_transform_function = label_transform_function
        for i, sec in enumerate(get_section):
            data_sample = data[sec]
            label_sample = label[sec]
            data_sample, label_sample = label_transform_function(data_sample, label_sample)
            data_sample, label_sample = data_transform_function(data_sample, label_sample)
            if i == 0:
                self.data_full = data_sample
                self.label_full = label_sample
            else:
                self.data_full = np.concatenate((self.data_full, data_sample), axis=-1)
                self.label_full = np.concatenate((self.label_full, label_sample), axis=-1)
        self.full_size = np.shape(self.label_full)
        self.out_size = 32

    def __getitem__(self, index):
        d_patches = patch_splitter_pixel(index, self.data_full, self.out_size, self.full_size)
        l_patches = patch_splitter_pixel(index, self.label_full, self.out_size, self.full_size)
        return d_patches, l_patches

    def __len__(self):
        return (self.full_size[1] - self.out_size + 1) * (self.full_size[0] - self.out_size + 1)


class DatasetGrid():
    def __init__(self, sampler_test,
                 window_size,
                 frequencies,
                 data_transform_function=None):
        self.sampler_test = sampler_test
        self.window_size = window_size
        self.frequencies = frequencies
        self.data_transform_function = data_transform_function

    def __getitem__(self, index):
        cumsum = np.cumsum(self.sampler_test.n_samples_per_echogram)
        cumsum_sub_idx = cumsum - index
        e_idx = np.where(cumsum_sub_idx > 0)[0][0]
        sub_idx = cumsum_sub_idx[e_idx]
        echogram = self.sampler_test.echograms[e_idx]
        center_location = self.sampler_test.center_locations[e_idx][-sub_idx]
        data, labels = get_crop(echogram, center_location, self.window_size, self.frequencies)

        # Apply data-transform-function
        if self.data_transform_function is not None:
            data = self.data_transform_function(data)
        labels = labels.astype('int16')
        return data, labels

    def __len__(self):
        return self.sampler_test.n_samples

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

    # labels = nearest_interpolation(echogram.label_memmap(), grid, boundary_val=-100, out_shape=window_size)
    labels = nearest_interpolation(echogram.label_memmap(), grid, boundary_val=-100, out_shape=window_size)

    return channels, labels
