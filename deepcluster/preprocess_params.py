
# Binary segmentation (background, sandeel)
'''
def preprocess_params():
    k = [2.019e-04, 2.506e-04, 1.781e-04, 1.292e-04]
    a = [-0.779, -0.812, -0.681, -0.578]
    params = {'k': k, 'a': a}
    return params
'''

# Segmentation with three classes (background, sandeel, other fish)
def preprocess_params():
    k = [4.153e-04, 3.464e-04, 2.692e-04, 2.613e-04]
    a = [-0.669, -1.115, -0.693, -0.593]

    params = {'k': k, 'a': a}
    return params