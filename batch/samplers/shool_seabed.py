import numpy as np

from batch.samplers.shool import Shool


class ShoolSeabed():
    def __init__(self, echograms, max_dist_to_seabed, fish_type='all'):
        """

        :param echograms: A list of all echograms in set
        """
        self.echograms = echograms

        #Get shools:
        self.shools = Shool(echograms, fish_type).shools

        #Remove shools that are not close to seabed
        self.shools = \
            [(e, o) for e, o in self.shools if
             np.abs(e.get_seabed()[int((o['bounding_box'][2] + o['bounding_box'][3]) / 2)] - o['bounding_box'][1]) <
             max_dist_to_seabed]

    def get_sample(self):
        """

        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        #Random object

        oi = np.random.randint(len(self.shools))
        e,o  = self.shools[oi]

        #Random pixel in object
        pi = np.random.randint(o['n_pixels'])
        y,x = o['indexes'][pi,:]

        #Todo: Call get_sample again if window does not contain seabed

        return [y,x], e