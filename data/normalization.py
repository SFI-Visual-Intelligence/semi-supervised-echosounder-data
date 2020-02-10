import numpy as np
from preprocess_params import preprocess_params

def db(data, *args):
    return 10 * np.log10(data + 1e-10)

def standardize_log_normal(batch):
    for f in range(batch.shape[1]):
        batch[:, f, :, :] -= np.nanmin(batch[:, f, :, :])
        batch[:, f, :, :] = np.log(batch[:, f, :, :] + 1e-10)
        batch[:, f, :, :] -= np.mean(batch[:, f, :, :])
        batch[:, f, :, :] *= 1 / (np.std(batch[:, f, :, :]) + 1e-10)
    return batch

def sigmoid_log_visualize_echogram(data, frequencies):
    if len(data.shape) != 3:
        print('sigmoid_log_visualize_echogram function requires that the number of input dimensions is 3. ', len(data.shape), ' were given.')
        return None
    elif data.shape[2] != 4:
        print('sigmoid_log_visualize_echogram function requires 4 input frequency channels. ', data.shape[2], ' were given.')
        if frequencies != [18, 38, 120, 200]:
            print('visualize function must use input parameter frequencies == [18, 38, 120, 200]. ', frequencies, ' were given.')
        return None
    else:
        eps = 1e-25
        k = np.array(preprocess_params()['k']).reshape(1, 1, 4)
        a = np.array(preprocess_params()['a']).reshape(1, 1, 4)
        data += eps
        data = 1 / (1 + k * np.power(data, a))
        return data