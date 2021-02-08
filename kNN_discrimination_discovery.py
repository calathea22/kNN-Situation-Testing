from scipy.spatial.distance import pdist, squareform, cdist
from optimize_euclidean_distances import weighted_euclidean_distance, luong_distance
from optimize_mahalanobis_distances import mahalanobis_distance
import pandas as pd
import numpy as np
import copy

def make_distance_row_luong(training_set, instance, indices_info):
    dist_row = cdist(instance.values[None,], training_set.values, luong_distance, indices_info=indices_info)
    return dist_row[0]


def find_2k_nearest_neighbors_Luong(k, instance, training_set, protected_indices, unprotected_indices, indices_info):
    distance_row = pd.Series(make_distance_row_luong(training_set, instance, indices_info))

    protected_instances = distance_row.iloc[protected_indices]
    unprotected_instances = distance_row.iloc[unprotected_indices]

    protected_neighbours_idx = np.argpartition(protected_instances, k)
    unprotected_neighbours_idx = np.argpartition(unprotected_instances, k)

    protected_neighbours = (protected_instances.iloc[protected_neighbours_idx[:k]])
    unprotected_neighbours = (unprotected_instances.iloc[unprotected_neighbours_idx[:k]])

    return (protected_neighbours, unprotected_neighbours)


def find_k_nearest_unprotected_neighbors_Luong(k, instance, training_set, unprotected_indices, indices_info):
    distance_row = pd.Series(make_distance_row_luong(training_set, instance, indices_info))

    unprotected_instances = distance_row.iloc[unprotected_indices]

    unprotected_neighbours_idx = np.argpartition(unprotected_instances, k)

    nearest_unprotected_neighbours = (unprotected_instances.iloc[unprotected_neighbours_idx[:k]])

    return nearest_unprotected_neighbours


def calc_difference(protected_neighbours, unprotected_neighbours, class_info):
    proportion_positive_protected = sum(class_info.iloc[protected_neighbours.index]) / len(protected_neighbours)
    proportion_positive_unprotected = sum(class_info.iloc[unprotected_neighbours.index]) / len(unprotected_neighbours)
    return (proportion_positive_unprotected - proportion_positive_protected)


def give_disc_label(discrimination_scores, threshold):
    discrimination_labels = []
    for discrimination_score in discrimination_scores:
        discrimination_labels.append(discrimination_score >= threshold)
    return discrimination_labels


def give_all_disc_scores_Luong(k, class_info_train, protected_indices_train, unprotected_indices_train, training_set, protected_indices_test, class_info_test, test_set, indices_info):
    discrimination_scores = []
    for protected_instance in protected_indices_test:
        if class_info_test.iloc[protected_instance] == 0:
            test_instance = test_set.iloc[protected_instance]
            protected_neighbours, unprotected_neighbours = find_2k_nearest_neighbors_Luong(k, test_instance,
                                                                                           training_set,
                                                                                           protected_indices_train,
                                                                                           unprotected_indices_train, indices_info)
            diff = calc_difference(protected_neighbours, unprotected_neighbours, class_info_train)
            discrimination_scores.append(diff)

        else:
            discrimination_scores.append(-1)
    return discrimination_scores


def give_all_disc_scores_Luong_unprotected_group(k, class_info_train, unprotected_indices_train, training_set, unprotected_indices_test, class_info_test, test_set, indices_info):
    discrimination_scores = []
    for unprotected_instance in unprotected_indices_test:
        if class_info_test.iloc[unprotected_instance] == 0:
            test_instance = test_set.iloc[unprotected_instance]
            nearest_unprotected_neighbours_and_their_distances = find_k_nearest_unprotected_neighbors_Luong(k, test_instance,
                                                                                                training_set,
                                                                                                unprotected_indices_train,
                                                                                                indices_info)
            class_nearest_neighbours = class_info_train.iloc[nearest_unprotected_neighbours_and_their_distances.index]
            discrimination_score = sum(class_nearest_neighbours) / len(class_nearest_neighbours)
            discrimination_scores.append(discrimination_score)
        else:
            discrimination_scores.append(-1)
    return discrimination_scores


def make_distance_row_euclidean(training_set, instance, indices_info, weights):
    dist_row = cdist(instance.values[None,], training_set.values, weighted_euclidean_distance, weights=weights, indices_info=indices_info)
    return dist_row[0]


def find_k_nearest_neighbors_euclidean(k, instance, training_set, indices_info, weights):
    distance_row = pd.Series(make_distance_row_euclidean(training_set, instance, indices_info, weights))
    nearest_neighbours_idx = np.argpartition(distance_row, k)

    nearest_neighbours = (distance_row.iloc[nearest_neighbours_idx[:k]])
    return nearest_neighbours


def find_k_nearest_unprotected_neighbors_euclidean(k, instance, training_set, unprotected_indices, indices_info, weights):
    distance_row = pd.Series(make_distance_row_euclidean(training_set, instance, indices_info, weights))

    unprotected_instances = distance_row.iloc[unprotected_indices]

    unprotected_neighbours_idx = np.argpartition(unprotected_instances, k)

    nearest_unprotected_neighbours = (unprotected_instances.iloc[unprotected_neighbours_idx[:k]])

    return nearest_unprotected_neighbours


def give_all_disc_scores_euclidean(k, class_info_train, protected_indices_train, unprotected_indices_train, training_set, protected_indices_test, class_info_test, test_set, indices_info, weights):
    discrimination_scores = []
    protected_instance_info = []
    for protected_instance in protected_indices_test:
        test_instance = test_set.iloc[protected_instance]
        print("_______________________________________________________")
        print(protected_instance)
        print(test_instance)
        if class_info_test.iloc[protected_instance] == 0:
            # protected_neighbours, unprotected_neighbours = find_2k_nearest_neighbors_euclidean(k, test_instance,
            #                                                                                training_set,
            #                                                                                protected_indices_train,
            #                                                                                unprotected_indices_train,
            #                                                                                indices_info, weights)
            # diff = calc_difference(protected_neighbours, unprotected_neighbours, class_info_train)
            # discrimination_scores.append(diff)

            nearest_neighbours_and_their_distances = find_k_nearest_unprotected_neighbors_euclidean(k, test_instance,
                                                                                          training_set, unprotected_indices_train,
                                                                                          indices_info, weights)
            print(nearest_neighbours_and_their_distances)
            # amount_of_protected_nearest_neighbours = len((set(nearest_neighbours_and_their_distances.index) & set(protected_indices_train)))
            # amount_of_protected_nearest_neighbours_list.append(amount_of_protected_nearest_neighbours)
            class_nearest_neighbours = class_info_train.iloc[nearest_neighbours_and_their_distances.index]
            discrimination_score = sum(class_nearest_neighbours) / len(class_nearest_neighbours)
            discrimination_scores.append(discrimination_score)
            protected_instance_info.append((protected_instance, 0))
        else:
            protected_instance_info.append((protected_instance, 1))
            discrimination_scores.append(-1)
            # amount_of_protected_nearest_neighbours_list.append(-1000)

    return discrimination_scores


def give_all_disc_scores_euclidean_unprotected_group(k, class_info_train, unprotected_indices_train, training_set, unprotected_indices_test, class_info_test, test_set, indices_info, weights):
    discrimination_scores = []
    for unprotected_instance in unprotected_indices_test:
        test_instance = test_set.iloc[unprotected_instance]
        if class_info_test.iloc[unprotected_instance] == 0:
            nearest_unprotected_neighbours_and_their_distances = find_k_nearest_unprotected_neighbors_euclidean(k, test_instance, training_set,
                                                                                                                unprotected_indices_train, indices_info, weights)

            class_nearest_neighbours = class_info_train.iloc[nearest_unprotected_neighbours_and_their_distances.index]
            discrimination_score = sum(class_nearest_neighbours) / len(class_nearest_neighbours)
            discrimination_scores.append(discrimination_score)
        else:
            discrimination_scores.append(-1)
    return discrimination_scores


def make_distance_row_mahalanobis(training_set, instance, indices_info, weight_matrix):
    dist_row = cdist(instance.values[None,], training_set.values, mahalanobis_distance, weight_matrix=weight_matrix,
                     indices_info=indices_info)
    return dist_row[0]


def find_k_nearest_neighbors_mahalanobis(k, instance, training_set, indices_info, weight_matrix):
    distance_row = pd.Series(make_distance_row_mahalanobis(training_set, instance, indices_info, weight_matrix))
    nearest_neighbours_idx = np.argpartition(distance_row, k)

    nearest_neighbours = (distance_row.iloc[nearest_neighbours_idx[:k]])
    return nearest_neighbours



def find_k_nearest_unprotected_neighbors_mahalanobis(k, instance, training_set, unprotected_indices, indices_info, weight_matrix):
    distance_row = pd.Series(make_distance_row_mahalanobis(training_set, instance, indices_info, weight_matrix))

    unprotected_instances = distance_row.iloc[unprotected_indices]

    unprotected_neighbours_idx = np.argpartition(unprotected_instances, k)

    nearest_unprotected_neighbours = (unprotected_instances.iloc[unprotected_neighbours_idx[:k]])

    return nearest_unprotected_neighbours



def give_all_disc_scores_mahalanobis(k, class_info_train, protected_indices_train, unprotected_indices_train, training_set, protected_indices_test, class_info_test, test_set, indices_info, weight_matrix):
    discrimination_scores = []
    for protected_instance in protected_indices_test:
        test_instance = test_set.iloc[protected_instance]
        print("_______________________________________________________")
        print(protected_instance)
        print(test_instance)
        if class_info_test.iloc[protected_instance] == 0:
            nearest_neighbours_and_their_distances = find_k_nearest_unprotected_neighbors_mahalanobis(k, test_instance, training_set,
                                                                                                    unprotected_indices_train, indices_info, weight_matrix)
            print(nearest_neighbours_and_their_distances)
            class_nearest_neighbours = class_info_train.iloc[nearest_neighbours_and_their_distances.index]
            discrimination_score = sum(class_nearest_neighbours) / len(class_nearest_neighbours)
            discrimination_scores.append(discrimination_score)
        else:
            discrimination_scores.append(-1)
    return discrimination_scores


def give_all_disc_scores_mahalanobis_unprotected_group(k, class_info_train, unprotected_indices_train, training_set, unprotected_indices_test, class_info_test, test_set, indices_info, weight_matrix):
    discrimination_scores = []
    for unprotected_instance in unprotected_indices_test:
        test_instance = test_set.iloc[unprotected_instance]
        if class_info_test.iloc[unprotected_instance] == 0:
            nearest_unprotected_neighbours_and_their_distances = find_k_nearest_unprotected_neighbors_mahalanobis(k, test_instance, training_set,
                                                                                                                unprotected_indices_train, indices_info, weight_matrix)

            class_nearest_neighbours = class_info_train.iloc[nearest_unprotected_neighbours_and_their_distances.index]
            discrimination_score = sum(class_nearest_neighbours) / len(class_nearest_neighbours)
            discrimination_scores.append(discrimination_score)
        else:
            discrimination_scores.append(-1)
    return discrimination_scores
