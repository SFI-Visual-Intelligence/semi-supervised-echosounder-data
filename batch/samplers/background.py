import numpy as np
from utils.np import  getGrid, nearest_interpolation


class Background():
    def __init__(self, echograms, window_size):
        """

        :param echograms: A list of all echograms in set
        """
        self.echograms = echograms
        self.window_size = window_size


    def get_sample(self):
        """

        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        #Random echogram
        ei = np.random.randint(len(self.echograms))

        #Random x,y-loc above seabed
        x = np.random.randint(self.window_size[1]//2, self.echograms[ei].shape[1] - self.window_size[1]//2)
        # y = np.random.randint(0, self.echograms[ei].get_seabed()[x])
        y = np.random.randint(self.window_size[1], int(0.8 * self.echograms[ei].get_seabed()[x]))

        #Check if there is any fish-labels in crop
        grid = getGrid(self.window_size) + np.expand_dims(np.expand_dims([y,x], 1), 1)
        labels = nearest_interpolation(self.echograms[ei].label_memmap(), grid, boundary_val=0, out_shape=self.window_size)

        if np.any(labels != 0):
            return self.get_sample() #Draw new sample

        return [y,x], self.echograms[ei]

    def get_all_samples(self, num_samples=6001):
        """
        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        center_locations = []
        echograms = []
        for i in range(num_samples):
            [y, x], e = self.get_sample()
            center_locations.append([y, x])
            echograms.append(e)
        return center_locations, echograms
