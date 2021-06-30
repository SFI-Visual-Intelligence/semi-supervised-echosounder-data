import numpy as np

def index_0_1_27_for_comparisonP2(data, labels):
    '''
    Re-assign labels to: Background==0, Sandeel==1, Other==2 - all remaining are set to ignore_value.
    :param data:
    :param labels:
    :param echogram:
    :param ignore_val:
    :return:
    '''

    new_labels = np.zeros(labels.shape)
    new_labels[np.where(labels == 27)] = 1
    new_labels[np.where(labels == 1)] = 2

    return data, new_labels
