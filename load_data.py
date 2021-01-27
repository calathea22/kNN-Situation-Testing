import pandas as pd
import utils
import numpy as np


def load_data(location, train_test_or_val):
    data = pd.read_excel(
        "C:/Users/daphn/PycharmProjects/kNN Situation Testing/" + location + "/" + train_test_or_val + "/data.xlsx")
    standardized_data = pd.read_excel(
        "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/"+train_test_or_val+"/standardized_data.xlsx")
    class_label = pd.read_excel(
        "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/"+train_test_or_val+"/class_label.xlsx")
    protected_info = pd.read_excel(
        "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/"+train_test_or_val+"/protected_attribute.xlsx")
    discriminated_instances = pd.read_excel(
        "C:/Users/daphn/PycharmProjects/kNN Situation Testing/" + location + "/"+train_test_or_val+"/discriminated_instances.xlsx")

    data = data.drop('Unnamed: 0', axis='columns')
    class_label = class_label.drop('Unnamed: 0', axis='columns')[0]
    standardized_data = standardized_data.drop('Unnamed: 0', axis='columns')
    protected_info = np.array(protected_info.drop('Unnamed: 0', axis='columns'))
    discriminated_instances = discriminated_instances.drop('Unnamed: 0', axis='columns')[0]
    protected_indices = list(np.where(protected_info == 1)[0])
    ground_truth_all_labels = utils.discriminated_instances_to_label_array(protected_indices,
                                                                            discriminated_instances)
    return {'data': data, 'standardized_data': standardized_data, 'protected_info': protected_info,
            'class_label': class_label, 'discriminated_instances': list(discriminated_instances), 'ground_truth': ground_truth_all_labels}


def load_optimization_info(location):
    weights_euclidean = pd.read_excel(
        "C:/Users/daphn/PycharmProjects/kNN Situation Testing/" + location + "/euclidean_weights.xlsx")
    mahalanobis_matrix = 0

    indices_info = {'interval': [0, 1, 2], 'ordinal': [], 'nominal': []}

    weights_euclidean = weights_euclidean.drop('Unnamed: 0', axis='columns')[0]
    #mahalanobis_matrix = mahalanobis_matrix.drop('Unnamed: 0', axis='columns')[0]

    return {'weights_euclidean': weights_euclidean, 'mahalanobis_matrix': mahalanobis_matrix, 'indices_info': indices_info}


def split_discriminated_indices(test_indices, remainder_instances, discriminated_indices):
    test_discriminated_indices = []
    remainder_discriminated_indices = []

    index_counter = 0
    for test_index in test_indices:
        if test_index in discriminated_indices:
            test_discriminated_indices.append(index_counter)
        index_counter +=1

    index_counter = 0
    for index in remainder_instances:
        if index in discriminated_indices:
            remainder_discriminated_indices.append(index_counter)
        index_counter +=1

    return remainder_discriminated_indices, test_discriminated_indices


def split_test_sets(n_splits, location):
    np.random.seed(6)
    all_test_sets_info = []
    loaded_test_set = load_data(location, "test")

    complete_non_standardized_data = loaded_test_set['data']
    complete_standardized_data = loaded_test_set['standardized_data'].to_numpy()
    complete_protected_info = loaded_test_set['protected_info'].flatten()
    complete_class_labels = loaded_test_set['class_label']
    complete_discriminated_labels = loaded_test_set['discriminated_instances']

    n_samples_per_split = (complete_non_standardized_data.shape[0])/n_splits

    remainder_standardized_data = complete_standardized_data.copy()
    remainder_non_standardized_data = complete_non_standardized_data.copy()
    remainder_protected_info = complete_protected_info.copy()
    remainder_class_labels = complete_class_labels.copy()
    remainder_discriminated_indices = complete_discriminated_labels.copy()

    for i in range(n_splits):
        split_non_standardized_data = remainder_non_standardized_data.sample(n=int(n_samples_per_split))
        split_indices = split_non_standardized_data.index
        split_standardized_data = remainder_standardized_data[split_indices]
        split_protected_info = remainder_protected_info[split_indices]
        split_class_labels = remainder_class_labels[split_indices]

        remainder_non_standardized_data = remainder_non_standardized_data.drop(split_indices)
        remainder_standardized_data = np.delete(remainder_standardized_data, split_indices, axis=0)
        remainder_protected_info = np.delete(remainder_protected_info, split_indices)
        remainder_class_labels = remainder_class_labels.drop(split_indices)

        remainder_indices = remainder_non_standardized_data.index

        remainder_discriminated_indices, split_discriminated= split_discriminated_indices(
            split_indices, remainder_indices, remainder_discriminated_indices)

        split_protected_indices = list(np.where(split_protected_info == 1)[0])
        split_ground_truth = utils.discriminated_instances_to_label_array(split_protected_indices,
                                                                           split_discriminated)
        remainder_non_standardized_data.reset_index(drop=True, inplace=True)
        split_non_standardized_data.reset_index(drop=True, inplace=True)
        remainder_class_labels.reset_index(drop=True, inplace=True)
        split_class_labels.reset_index(drop=True, inplace=True)

        test_info = {'data': split_non_standardized_data,'standardized_data': pd.DataFrame(split_standardized_data), 'protected_info': split_protected_info,
        'class_label': split_class_labels, 'discriminated_indices': split_discriminated, 'ground_truth': split_ground_truth}
        all_test_sets_info.append(test_info)

    return all_test_sets_info