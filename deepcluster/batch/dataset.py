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
        self.n_samples = len(data)
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function


    def __getitem__(self, index):
        data_sample = self.data[index]
        label_sample = self.label[index]

        # Apply data-transform-function
        if self.label_transform_function is not None:
            data_sample, label_sample_t = self.label_transform_function(data_sample, label_sample)
        # Apply label_augmentation
        if self.data_transform_function is not None:
            data_sample_t, label_sample_t = self.data_transform_function(data_sample, label_sample_t)

        return data_sample_t, label_sample_t, index



    def __len__(self):
        return self.n_samples


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

# class DatasetSampler():
#     def __init__(self, sampler, window_size, frequencies):
#         self.sampler = sampler
#         self.window_size = window_size
#         self.frequencies = frequencies
#
#     def __getitem__(self, index):
#         #Select which sampler to use
#         #Draw coordinate and echogram with sampler
#         center_locations, echograms = self.sampler.get_all_samples()
#         #Get data/labels-patches
#         for i, (echogram, center_location) in enumerate(zip(echograms, center_locations)):
#             data, _ = get_crop(echogram, center_location, self.window_size, self.frequencies)
#             yield data

# def sampling_echograms_full(window_size, args):
#     path_to_echograms = paths.path_to_echograms()
#     with open(os.path.join(path_to_echograms, 'memmap_2014_heave.pkl'), 'rb') as fp:
#         eg_names_full = pickle.load(fp)
#     echograms = get_echograms_full(eg_names_full)
#     echograms_train, echograms_val, echograms_test = cps.partition_data(echograms, args.partition, portion_train_test=0.8, portion_train_val=0.75)
#
#     sampler_bg_train = Background(echograms_train, window_size)
#     sampler_sh27_train = Shool(echograms_train, window_size, 27)
#     sampler_sbsh27_train = ShoolSeabed(echograms_train, window_size, args.window_dim//4, fish_type=27)
#     sampler_sh01_train = Shool(echograms_train, window_size, 1)
#     sampler_sbsh01_train = ShoolSeabed(echograms_train, window_size, args.window_dim//4, fish_type=1)
#
#     samplers_train = [sampler_bg_train,
#                       sampler_sh27_train, sampler_sbsh27_train,
#                       sampler_sh01_train, sampler_sbsh01_train]
#
#     augmentation = CombineFunctions([add_noise, flip_x_axis])
#     label_transform = CombineFunctions([index_0_1_27, relabel_with_threshold_morph_close])
#     data_transform = CombineFunctions([remove_nan_inf, db_with_limits])
#
#     dataset_train = Dataset(
#         samplers_train,
#         window_size,
#         args.frequencies,
#         args.batch * args.iteration_train,
#         args.sampler_probs,
#         augmentation_function=augmentation,
#         label_transform_function=label_transform,
#         data_transform_function=data_transform)
#
#     return dataset_train

# class DatasetVal():
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
#         return (data, sample_idx)
#
#     def __len__(self):
#         return self.n_samples
