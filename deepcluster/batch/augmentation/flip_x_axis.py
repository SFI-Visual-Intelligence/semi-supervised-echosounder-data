import numpy as np
import copy

def flip_x_axis_img(data):
    if np.random.randint(2):
        data = copy.copy(np.flip(data, 2))
    return data

# def flip_x_axis(data, labels, echogram):
#     if np.random.randint(2):
#         data = np.flip(data, 2).copy()
#         labels = np.flip(labels, 1).copy()
#     return data, labels, echogram
