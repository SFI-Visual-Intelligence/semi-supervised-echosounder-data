import numpy as np

def remove_nan_inf(data, labels, echogram, frequencies, new_value=0.0):
    '''
    Reassigns all non-finite data values (nan, positive inf, negative inf) to new_value.
    :param data:
    :param labels:
    :param echogram:
    :param new_value:
    :return:
    '''
    data[np.invert(np.isfinite(data))] = new_value
    return data, labels, echogram, frequencies