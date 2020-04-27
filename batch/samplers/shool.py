import numpy as np
from utils.np import  getGrid, nearest_interpolation

class Shool():
    def __init__(self, echograms, window_size, fish_type):
        """
        :param echograms: A list of all echograms in set
        """
        self.window_size = window_size
        self.echograms = echograms
        self.fish_type = fish_type
        self.solid_shools = []
        self.seabed_shools = []

        #Remove echograms without fish
        if self.fish_type == 'all':
            self.echograms = [e for e in self.echograms if len(e.objects)>0]
            for e in self.echograms:
                for o in e.objects:
                    mean_yseabed = np.mean(e.get_seabed()[np.unique(o['indexes'][:, 1])])
                    [mean_y, _] = o['indexes'].mean(axis=0)
                    if (mean_y + self.window_size[0] // 2) < mean_yseabed:
                        self.solid_shools.append((e, o))
                    else:
                        self.seabed_shools.append((e, o))

        elif type(self.fish_type) == int:
            self.echograms = [e for e in self.echograms if any([o['fish_type_index'] == self.fish_type for o in e.objects])]
            for e in self.echograms:
                for o in e.objects:
                    if o['fish_type_index'] == self.fish_type:
                        mean_yseabed = np.mean(e.get_seabed()[np.unique(o['indexes'][:, 1])])
                        [mean_y, _] = o['indexes'].mean(axis=0)
                        if (mean_y + self.window_size[0] // 2) < mean_yseabed:
                            self.solid_shools.append((e, o))
                        else:
                            self.seabed_shools.append((e, o))

        elif type(self.fish_type) == list:
            self.echograms = [e for e in self.echograms if any([o['fish_type_index'] in self.fish_type for o in e.objects])]
            for e in self.echograms:
                for o in e.objects:
                    if o['fish_type_index'] in self.fish_type:
                        mean_yseabed = np.mean(e.get_seabed()[np.unique(o['indexes'][:, 1])])
                        [mean_y, _] = o['indexes'].mean(axis=0)
                        if (mean_y + self.window_size[0] // 2) < mean_yseabed:
                            self.solid_shools.append((e, o))
                        else:
                            self.seabed_shools.append((e, o))
        else:
            class UnknownFishType(Exception):pass
            raise UnknownFishType('Should be int, list of ints or "all"')

        if len(self.echograms) == 0:
            class EmptyListOfEchograms(Exception):pass
            raise EmptyListOfEchograms('fish_type not found in any echograms')


    def get_sample(self):

        """
        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        #Random object
        oi = np.random.randint(len(self.solid_shools))
        e,o = self.solid_shools[oi]

        #Random pixel in object
        pi = np.random.randint(o['n_pixels'])
        y,x = o['indexes'][pi, :]

        # Correct x if window is not inside echogram
        if (x < self.window_size[1]//2):
            x = self.window_size[1]//2
        elif (x > e.shape[1] - self.window_size[1]//2):
            x = e.shape[1] - self.window_size[1]//2
        return [y,x], e

    def get_all_samples(self):
        """
        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        center_locations = []
        echograms = []
        #Random object
        for i, (e, o) in enumerate(self.solid_shools):
        #Random pixel in object
            pi = np.random.randint(o['n_pixels'])
            y,x = o['indexes'][pi, :]
            # Correct x if window is not inside echogram
            if (x < self.window_size[1]//2):
                x = self.window_size[1]//2
            elif (x > e.shape[1] - self.window_size[1]//2):
                x = e.shape[1] - self.window_size[1]//2
            center_locations.append([y, x])
            echograms.append(e)
        return center_locations, echograms

