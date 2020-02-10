import numpy as np

class Shool():
    def __init__(self, echograms, fish_type='all'):
        """

        :param echograms: A list of all echograms in set
        """

        self.echograms = echograms
        self.fish_type = fish_type

        self.shools = []
        #Remove echograms without fish
        if self.fish_type == 'all':
            self.echograms = [e for e in self.echograms if len(e.objects)>0]
            for e in self.echograms:
                for o in e.objects:
                    self.shools.append((e,o))

        elif type(self.fish_type) == int:
            self.echograms = [e for e in self.echograms if any([o['fish_type_index'] == self.fish_type for o in e.objects])]
            for e in self.echograms:
                for o in e.objects:
                    if o['fish_type_index'] == self.fish_type:
                        self.shools.append((e,o))

        elif type(self.fish_type) == list:
            self.echograms = [e for e in self.echograms if any([o['fish_type_index'] in self.fish_type for o in e.objects])]
            for e in self.echograms:
                for o in e.objects:
                    if o['fish_type_index'] in self.fish_type:
                        self.shools.append((e,o))

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

        oi = np.random.randint(len(self.shools))
        e,o  = self.shools[oi]

        #Random pixel in object
        pi = np.random.randint(o['n_pixels'])
        y,x = o['indexes'][pi,:]

        return [y,x], e
