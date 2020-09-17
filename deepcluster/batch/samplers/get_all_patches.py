import torch
import numpy as np
from utils.np import getGrid, linear_interpolation, nearest_interpolation

class GetAllPatches():
    def __init__(self, echograms, window_size, stride, fish_type, random_offset_ratio, phase:str):
        self.window_size = window_size
        self.stride = stride
        self.echograms = echograms
        self.fish_type = fish_type
        self.surface_offset_pixel = 10
        self.color_mapping = {0: (1, 1, 1), 1:(1, 0 ,0), 27:(0, 0, 1), 'others':(0, 1, 0)}
        self.random_offset_ratio = random_offset_ratio
        self.phase = phase
        self.target_echograms = self.get_target_echogram()

    def get_target_echogram(self):
        if self.fish_type == 'all':
            target_echograms = [e for e in self.echograms if len(e.objects)>0]

        elif type(self.fish_type) == int:
            target_echograms = [e for e in self.echograms if any([o['fish_type_index'] == self.fish_type for o in e.objects])]

        elif type(self.fish_type) == list:
            target_echograms = [e for e in self.echograms if any([o['fish_type_index'] in self.fish_type for o in e.objects])]

        else:
            class UnknownFishType(Exception):pass
            raise UnknownFishType('Should be int, list of ints or "all"')

        if len(target_echograms) == 0:
            class EmptyListOfEchograms(Exception):pass
            raise EmptyListOfEchograms('fish_type not found in any echograms')

        if self.phase == 'eval':
            temp_target_e = []
            for e in target_echograms:
                count = 0
                for o in e.objects:
                    if o['fish_type_index'] == 27:
                        count += 1
                if count > 10:
                    temp_target_e.append(e)
            target_echograms = temp_target_e
        print('num_target_echo: ', len(target_echograms))
        return target_echograms

    def gen_patch(self):
        self.bg = []
        self.fish_01 = []
        self.fish_27 = []
        self.bg_sb = []
        self.fish_01_sb = []
        self.fish_27_sb = []

        self.fish_01_label = []
        self.fish_27_label = []
        self.fish_01_sb_label = []
        self.fish_27_sb_label = []

        self.mixed_shool = []
        self.mixed_shool_sb = []
        self.mixed_shool_label = []
        self.mixed_shool_sb_label = []

        self.non_use = []
        self.non_use_label = []

        for e in self.target_echograms:
            center_locations = self.get_center_locations(e, self.window_size, self.stride, self.surface_offset_pixel, self.random_offset_ratio)
            bg_e = []
            fish_01_e = []
            fish_27_e = []
            bg_sb_e = []
            fish_01_sb_e = []
            fish_27_sb_e = []

            fish_01_label_e = []
            fish_27_label_e = []
            fish_01_sb_label_e = []
            fish_27_sb_label_e = []

            for center_location in center_locations:
                seabed_flag = False
                yrange = range(center_location[0]-self.window_size[0]//2, center_location[0]+self.window_size[0]//2)
                xrange = range(center_location[1]-self.window_size[1]//2, center_location[1]+self.window_size[1]//2)
                sb_y = e.get_seabed()[xrange]
                if np.array([sb_y[i] in yrange for i in range(len(sb_y))]).any():
                    seabed_flag = True
                    new_yrange = range(yrange[0]+self.window_size[0]//8, (yrange[-1]+1)-self.window_size[0]//8)
                    if not np.array([sb_y[i] in new_yrange for i in range(len(sb_y))]).any():  # ignore patches with marginal sb features
                        continue
                channels, label = self.get_crop(e, center_location, self.window_size)
                class_in_patch, count_pixels = self.get_class_in_patch(label)
                label_decision = self.get_label_decision(class_in_patch, count_pixels)

                label_plot = np.zeros([self.window_size[0], self.window_size[1], 3])
                for cl in class_in_patch:
                    if cl in ([0] + self.fish_type):
                        label_plot[label == cl] = self.color_mapping[cl]
                    else:
                        label_plot[label == cl] = self.color_mapping['others']

                if (len(label_decision) == 1) and (label_decision[0] in ([0] + self.fish_type)):
                    if seabed_flag == False:
                        if label_decision[0] == 0:
                            bg_e.append(channels)
                        elif label_decision[0] == 1:
                            fish_01_e.append(channels)
                            fish_01_label_e.append(label_plot)
                        else:
                            fish_27_e.append(channels)
                            fish_27_label_e.append(label_plot)
                    else: # seabed flag: True
                        if label_decision[0] == 0:
                            bg_sb_e.append(channels)
                        elif label_decision[0] == 1:
                            fish_01_sb_e.append(channels)
                            fish_01_sb_label_e.append(label_plot)
                        else:
                            fish_27_sb_e.append(channels)
                            fish_27_sb_label_e.append(label_plot)

                elif len(label_decision) > 1:
                    if seabed_flag == False:
                        self.mixed_shool.append(channels)
                        self.mixed_shool_label.append(label_plot)
                    else:
                        self.mixed_shool_sb.append(channels)
                        self.mixed_shool_sb_label.append(label_plot)

                else:
                    self.non_use.append(channels)
                    self.non_use_label.append(label_plot)

            if self.phase == 'train':
                bg_size = (len(fish_01_e) + len(fish_27_e)) // 2
                if bg_size > 0:
                    bg_idx = np.random.choice(np.arange(len(bg_e)), bg_size)
                    bg_e = [bg_e[i] for i in bg_idx]
                    self.bg.extend(bg_e)

                bg_sb_size = (len(fish_01_sb_e)+ len(fish_27_sb_e)) // 2
                if bg_sb_size > 0:
                    bg_sb_idx = np.random.choice(np.arange(len(bg_sb_e)), bg_sb_size)
                    bg_sb_e = [bg_sb_e[i] for i in bg_sb_idx]
                    self.bg_sb.extend(bg_sb_e)

                self.fish_01.extend(fish_01_e)
                self.fish_27.extend(fish_27_e)
                self.fish_01_sb.extend(fish_01_sb_e)
                self.fish_27_sb.extend(fish_27_sb_e)
                self.fish_01_label.extend(fish_01_label_e)
                self.fish_27_label.extend(fish_27_label_e)
                self.fish_01_sb_label.extend(fish_01_sb_label_e)
                self.fish_27_sb_label.extend(fish_27_sb_label_e)

        print('#############   ', self.phase, '#############     \n', self.bg.__len__(), '\t', self.fish_01.__len__(),  '\t',self.fish_27.__len__(), '\n',
              self.bg_sb.__len__(),  '\t',self.fish_01_sb.__len__(),  '\t',self.fish_27_sb.__len__(),  '\n',
              self.mixed_shool.__len__(),  '\t',self.mixed_shool_sb.__len__(),  '\t',self.non_use.__len__())

        torch.save(self.bg, 'bg.pt')
        torch.save(self.fish_01, 'fish01.pt')
        torch.save(self.fish_01_label, 'fish01lb.pt')
        torch.save(self.fish_27, 'fish27.pt')
        torch.save(self.fish_27_label, 'fish27lb.pt')
        torch.save(self.bg_sb, 'bgsb.pt')
        torch.save(self.fish_01_sb, 'fish01sb.pt')
        torch.save(self.fish_01_sb_label, 'fish01sblb.pt')
        torch.save(self.fish_27_sb, 'fish27sb.pt')
        torch.save(self.fish_27_sb_label, 'fish27sblb.pt')

    @staticmethod
    def get_center_locations(e, window_size, stride, surface_offset_pixel, random_offset_ratio):  # random_offset: window_size/random offset_ratio
        ymax, xmax = e.shape  # (483, 3028)
        # nrow = (ymax - surface_offset_pixel - window_size[0] - window_size[0]//(random_offset_ratio//2))//stride[0] + 1
        # ncol = (xmax - window_size[1] - window_size[1]//(random_offset_ratio//2))//stride[1] + 1
        depths = np.arange(surface_offset_pixel + window_size[1]//random_offset_ratio + window_size[1], ymax - window_size[1]//random_offset_ratio, stride[1]) - window_size[1]//2
        widths = np.arange(window_size[0]//random_offset_ratio + window_size[0], xmax - window_size[0]//random_offset_ratio, stride[0]) - window_size[0]//2
        y_locs, x_locs = np.meshgrid(depths, widths, indexing='xy')

        y_offset = np.random.randint(-window_size[1]//random_offset_ratio, window_size[1]//random_offset_ratio, size=y_locs.shape)
        x_offset = np.random.randint(-window_size[0]//random_offset_ratio, window_size[0]//random_offset_ratio, size=x_locs.shape)

        y_locs = y_locs + y_offset
        x_locs = x_locs + x_offset

        y_locs = y_locs.ravel()
        x_locs = x_locs.ravel()

        center_locations = []
        for (y, x) in zip(y_locs, x_locs):
            center_locations.append([y, x])
        return center_locations

    @staticmethod
    def get_crop(e, center_location, window_size, freqs=[18, 38, 120, 200]):
        """
        Returns a crop of data around the pixels specified in the center_location.
        """
        # Get grid sampled around center_location
        grid = getGrid(window_size) + np.expand_dims(np.expand_dims(center_location, 1), 1)
        channels = []
        for f in freqs:
            # Interpolate data onto grid
            memmap = e.data_memmaps(f)[0]
            data = linear_interpolation(memmap, grid, boundary_val=0, out_shape=window_size)
            del memmap
            # Set non-finite values (nan, positive inf, negative inf) to zero
            if np.any(np.invert(np.isfinite(data))):
                data[np.invert(np.isfinite(data))] = 0
            channels.append(np.expand_dims(data, 0))
        channels = np.concatenate(channels, 0)
        labels = nearest_interpolation(e.label_memmap(), grid, boundary_val=-100, out_shape=window_size)
        return channels, labels

    @staticmethod
    def get_class_in_patch(labels):
        labels_vec = labels.ravel()
        class_in_patch = list(set(labels_vec))
        count_pixels = []
        for cl in class_in_patch:
            count_pixels.append(np.isin(labels_vec, cl).sum())
        return class_in_patch, count_pixels

    @staticmethod
    def get_label_decision(class_in_patch, count_pixels):
        size_descending_order = np.argsort(count_pixels)[::-1]
        class_descending_order = [class_in_patch[i] for i in size_descending_order]
        if len(class_descending_order) == 1:
            label_decision = [class_descending_order[0]]
        elif (len(class_descending_order) == 2) and (class_descending_order[0] == 0):
            label_decision = [class_descending_order[1]]
        elif (len(class_descending_order) == 2) and (class_descending_order[0] != 0):
            label_decision = [class_descending_order[0]]
        else:
            label_decision = class_descending_order
        return label_decision


