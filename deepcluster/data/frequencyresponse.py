import numpy as np
import matplotlib.pyplot as plt
import pdb


def plotfrequencyresponse(echogram, freqs, labels="All"):
    """ Plot frequency response curves
    echograms is a list of echograms
    freq is a list of frequencies
    classes is the classes to add to the plot"""

    # Get the frequencies and classes for all echograms
    
    #if freq == 'All':
    #    freqs = 5  # The intersect of freqs in echograms
    #data = self.data_numpy(frequencies)

    # List frequencies in all echograms

    # Only use echogram with selected frequencies AND that is in all echograms
    # echogram = [s for s in echogram if all([f in s.frequencies for f in freqs])]

    frequencyresponse = np.zeros([0, len(freqs)])

    for s in echogram:
        # the the frequency response from the echogram
        frequencyresponse_perechogram = getfreqresponse(s, freqs)
        np.append(frequencyresponse, frequencyresponse_perechogram, axis=0)


def getfreqresponse(echogram, freqs):
    # Get the data and labels

    data = echogram.data_numpy(frequencies=freqs)
    label = echogram.label_numpy()
    # Get unique labels
    r = np.zeros(data.shape)
    base_freq_ind = 2
    r_freqs_ind = (0, 1, 2, 3)
 
    for i in r_freqs_ind:
        nils = data[:, :, i]
        nils2 = data[:, :, base_freq_ind]
        r[:, :, i] = nils/nils2  # data[:, :, i]/data[:, :, base_freq_ind]

    #pdb.set_trace()
    # number of frequency responses
    n = data.shape

    
    # Calculate r
    # r = np.zeros(
    #frequencyresponse = np.zeros
