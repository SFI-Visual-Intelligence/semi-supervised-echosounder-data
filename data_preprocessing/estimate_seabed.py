
import os

import paths
from data.echogram import Echogram


def save_all_seabeds():
    """
    Loop through all echograms and generate seabed-estimates
    :return:
    """
    path_to_echograms = paths.path_to_echograms()
    echogram_names = os.listdir(path_to_echograms)
    echograms = [Echogram(path_to_echograms + e) for e in echogram_names]
    for e in echograms:
        e.get_seabed(save_to_file=True, ignore_saved=True)


if __name__ == '__main__':
    save_all_seabeds()