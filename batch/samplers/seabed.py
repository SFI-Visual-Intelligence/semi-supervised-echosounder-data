import numpy as np
from utils.np import  getGrid, nearest_interpolation

class Seabed():
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

        #Random x-loc
        x = np.random.randint(self.window_size[1]//2, self.echograms[ei].shape[1] - (self.window_size[1]//2))
        y = self.echograms[ei].get_seabed()[x] + np.random.randint(-self.window_size[0]//2, self.window_size[0]//2)
        # y = self.echograms[ei].get_seabed()[x] + np.random.randint(-self.window_size[0], self.window_size[0])
        # y = self.echograms[ei].get_seabed()[x]

        # Correct y if window is not inside echogram
        if y < self.window_size[0]//2:
            y = self.window_size[0]//2
        if y > self.echograms[ei].shape[0] - self.window_size[0]//2:
            y = self.echograms[ei].shape[0] - self.window_size[0]//2

        #Check if there is any fish-labels in crop
        grid = getGrid(self.window_size) + np.expand_dims(np.expand_dims([y,x], 1), 1)
        labels = nearest_interpolation(self.echograms[ei].label_memmap(), grid, boundary_val=0, out_shape=self.window_size)

        if np.any(labels != 0):
            return self.get_sample() #Draw new sample

        return [y,x], self.echograms[ei]
