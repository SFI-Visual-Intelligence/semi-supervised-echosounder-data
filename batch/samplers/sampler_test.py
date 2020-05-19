import numpy as np

class SampleFull():
    def __init__(self, echograms, window_size, stride):
        self.echograms = echograms
        self.window_size = window_size
        self.stride = stride
        self.center_locations, self.n_samples, self.n_samples_per_echogram = self.get_all_samples()

    def get_all_samples(self):
        """
        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        count = 0
        center_locations = []
        count_per_echogram = []
        for echo_idx, e in enumerate(self.echograms):
            dim = e.shape
            ys = np.arange(self.window_size[0]//2, dim[0]-self.window_size[0]//2, self.stride)
            xs = np.arange(self.window_size[1]//2, dim[1]-self.window_size[1]//2, self.stride)
            center_locations.append(self.get_full_idxes(ys, xs))
            count_echo = len(ys) * len(xs)
            count_per_echogram.append(count_echo)
            count += count_echo
        return center_locations, count, count_per_echogram

    @staticmethod
    def get_full_idxes(ys, xs):
        centroids = []
        for y in ys:
            for x in xs:
                centroids.append([y, x])
        return centroids



