import numpy as np

def remove_nan_inf_for_comparisonP2(data, labels):
    '''
    Reassigns all non-finite data values (nan, positive inf, negative inf) to new_value.
    :param data:
    :param labels:
    :param echogram:
    :param new_value:
    :return:
    '''
    data[np.invert(np.isfinite(data))] = -75
    return data, labels

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

def remove_nan_inf_img(data, new_value=0.0):
    '''
    Reassigns all non-finite data values (nan, positive inf, negative inf) to new_value.
    :param data:
    :param labels:
    :param echogram:
    :param new_value:
    :return:
    '''
    data[np.invert(np.isfinite(data))] = new_value
    return data