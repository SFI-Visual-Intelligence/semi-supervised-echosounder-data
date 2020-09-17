
import numpy as np
from data.echogram import get_echograms
from paths import path_to_echograms


def save_memmap(data, path, dtype):
    path = (path + '.dat').replace('.dat.dat','.dat')
    fp = np.memmap(path, dtype=dtype, mode='w+', shape=data.shape)
    fp[:] = data.astype(dtype)
    del fp


def generate_and_save_heave_files():

    root = path_to_echograms()

    echs = get_echograms()
    for i, ech in enumerate(echs):

        if i % 100 == 0:
            print(len(echs), i)

        # Get vertical pixel resolution
        r = ech.range_vector
        r_diff = np.median(r[1:] - r[:-1])

        # Convert heave value from meters to number of pixels
        heave = np.round(ech.heave / r_diff).astype(np.int)
        assert heave.size == ech.shape[1]

        labels_old = ech.label_numpy()
        labels_new = np.zeros_like(labels_old)

        # Create new labels: Move each labels column up/down corresponding to heave
        for x, h in enumerate(list(heave)):
            if h == 0:
                labels_new[:, x] = labels_old[:, x]
            elif h > 0:
                labels_new[:-h, x] = labels_old[h:, x]
            else:
                labels_new[-h:, x] = labels_old[:h, x]

        # Save new labels as new memmap file
        path_save = root + ech.name + '/labels_heave'
        save_memmap(labels_new, path_save, dtype=labels_new.dtype)


if __name__ == '__main__':

    ### Uncomment line below and run this script to save memmap label files with reversed heave compensation ###

    # generate_and_save_heave_files()
    pass



