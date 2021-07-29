import os
import paths
import torch
import torch.nn as nn
from batch.data_transform_functions.remove_nan_inf import remove_nan_inf_for_comparisonP2
from batch.data_transform_functions.db_with_limits import db_with_limits_for_comparisonP2
from batch.combine_functions import CombineFunctions
from batch.label_transform_functions.index_0_1_27_for_comparisonP2 import index_0_1_27_for_comparisonP2
from batch.label_transform_functions.relabel_with_threshold_morph_close_for_comparisonP2 import relabel_with_threshold_morph_close_for_comparisonP2
from batch.label_transform_functions.seabed_checker_for_comparisonP2 import seabed_checker_for_comparisonP2
from batch.dataset import DatasetImg_for_comparisonP2


def sampling_echograms_full_for_comparisonP2(args):
    path_to_echograms = paths.path_to_echograms()
    data = torch.load(os.path.join(path_to_echograms, 'data_tr_TEST_200.pt'))
    label = torch.load(os.path.join(path_to_echograms, 'label_tr_TEST_200.pt'))
    data_transform = CombineFunctions([remove_nan_inf_for_comparisonP2, db_with_limits_for_comparisonP2])
    label_transform = CombineFunctions([index_0_1_27_for_comparisonP2, relabel_with_threshold_morph_close_for_comparisonP2, seabed_checker_for_comparisonP2])

    semi_count = int(len(data) * args.semi_ratio)

    dataset_cp = DatasetImg_for_comparisonP2(data=data,
                                          label=label,
                                          label_transform_function=label_transform,
                                          data_transform_function=data_transform)

    dataset_semi = DatasetImg_for_comparisonP2(data=data[:semi_count],
                                          label=label[:semi_count],
                                          label_transform_function=label_transform,
                                          data_transform_function=data_transform)

    return dataset_cp, dataset_semi


def sampling_echograms_test_for_comparisonP2():
    path_to_echograms = paths.path_to_echograms()
    data = torch.load(os.path.join(path_to_echograms, 'data_te_TEST_60.pt'))
    label = torch.load(os.path.join(path_to_echograms, 'label_te_TEST_60.pt'))
    data_transform = CombineFunctions([remove_nan_inf_for_comparisonP2, db_with_limits_for_comparisonP2])
    label_transform = CombineFunctions([index_0_1_27_for_comparisonP2, relabel_with_threshold_morph_close_for_comparisonP2, seabed_checker_for_comparisonP2])

    dataset = DatasetImg_for_comparisonP2(data=data,
                                          label=label,
                                          label_transform_function=label_transform,
                                          data_transform_function=data_transform)
    return dataset


def sampling_echograms_2019_for_comparisonP2(echogram_idx=2, path_to_echograms=None):
    if path_to_echograms == None:
        path_to_echograms = paths.path_to_echograms()
    data, label, patch_loc = torch.load(os.path.join(path_to_echograms, 'data_label_patch_loc_te_2019_%d.pt' % echogram_idx))
    data_transform = CombineFunctions([remove_nan_inf_for_comparisonP2, db_with_limits_for_comparisonP2])
    label_transform = CombineFunctions([index_0_1_27_for_comparisonP2, relabel_with_threshold_morph_close_for_comparisonP2, seabed_checker_for_comparisonP2])

    dataset_2019 = DatasetImg_for_comparisonP2(data=data,
                                          label=label,
                                          label_transform_function=label_transform,
                                          data_transform_function=data_transform)
    return dataset_2019, patch_loc