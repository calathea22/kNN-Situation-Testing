from scipy.spatial.distance import cdist
from optimize_euclidean_distances import weighted_euclidean_distance, luong_distance
from optimize_mahalanobis_distances import mahalanobis_distance
import pandas as pd
import numpy as np


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


# def find_k_nearest_unprotected_neighbors_Luong(k, instance, training_set, unprotected_indices, indices_info):
#     distance_row = pd.Series(make_distance_row_luong(training_set, instance, indices_info))
#
#     unprotected_instances = distance_row.iloc[unprotected_indices]
#
#     unprotected_neighbours_idx = np.argpartition(unprotected_instances, k)
#
#     nearest_unprotected_neighbours = (unprotected_instances.iloc[unprotected_neighbours_idx[:k]])
#
#     return nearest_unprotected_neighbours


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


def find_k_nearest_unprotected_neighbors(k, instance, train_set_unprotected, indices_info, distance_function, weights):
    distance_row = pd.Series(cdist(instance.values[None,], train_set_unprotected.values, distance_function, weights= weights, indices_info=indices_info)[0])

    nearest_neighbours_idx = np.argpartition(distance_row, k)

    nearest_neighbours = (distance_row.iloc[nearest_neighbours_idx[:k]])

    distance_to_closest_nearest_neighbour = min(nearest_neighbours)

    return nearest_neighbours, distance_to_closest_nearest_neighbour


def give_decision_labels_unprotected_group(k, class_info_train, unprotected_indices_train, training_set, unprotected_indices_test, class_info_test, test_set, indices_info, distance_function, weights=None):
    predicted_scores_test = []
    train_set_unprotected = training_set.iloc[unprotected_indices_train].reset_index(drop=True)
    train_class_info_unprotected = class_info_train.iloc[unprotected_indices_train].reset_index(drop=True)
    for unprotected_index in unprotected_indices_test:
        test_instance = test_set.iloc[unprotected_index]
        nearest_unprotected_neighbours_and_their_distances, _ = find_k_nearest_unprotected_neighbors(k, test_instance, train_set_unprotected, indices_info, distance_function, weights)
        class_nearest_neighbours = train_class_info_unprotected.iloc[nearest_unprotected_neighbours_and_their_distances.index]
        ratio_of_positive_class_labels_neighbours = sum(class_nearest_neighbours) / k
        ratio_of_negative_class_labels_neighbours = 1 - ratio_of_positive_class_labels_neighbours
        predicted_scores_test.append(ratio_of_positive_class_labels_neighbours - ratio_of_negative_class_labels_neighbours)

    test_class_info_unprotected = class_info_test.iloc[unprotected_indices_test].reset_index(drop=True)
    predicted_scores_test = np.array(predicted_scores_test)

    negative_class_indices_test = np.where(test_class_info_unprotected==0)[0]
    positive_class_indices_test = np.where(test_class_info_unprotected==1)[0]
    predictions_negative_class = predicted_scores_test[negative_class_indices_test]
    print(predictions_negative_class)
    predictions_positive_class = predicted_scores_test[positive_class_indices_test]
    print(predictions_positive_class)

    #print("Accuracy score: " + str(accuracy_score(test_class_info_unprotected, predicted_labels_test)))
    return predictions_negative_class, predictions_positive_class


def make_distance_row_euclidean(training_set, instance, indices_info, weights):
    dist_row = cdist(instance.values[None,], training_set.values, weighted_euclidean_distance, weights=weights, indices_info=indices_info)
    return dist_row[0]


def find_2k_nearest_neighbors_euclidean(k, instance, training_set, protected_indices, unprotected_indices, indices_info, weights):
    distance_row = pd.Series(make_distance_row_euclidean(training_set, instance, indices_info, weights))
    protected_instances = distance_row.iloc[protected_indices]
    unprotected_instances = distance_row.iloc[unprotected_indices]

    protected_neighbours_idx = np.argpartition(protected_instances, k)
    unprotected_neighbours_idx = np.argpartition(unprotected_instances, k)

    protected_neighbours = (protected_instances.iloc[protected_neighbours_idx[:k]])
    unprotected_neighbours = (unprotected_instances.iloc[unprotected_neighbours_idx[:k]])

    return protected_neighbours, unprotected_neighbours



def give_all_disc_scores_euclidean_2k_approach(k, class_info_train, protected_indices_train, unprotected_indices_train, training_set, protected_indices_test, class_info_test, test_set, indices_info, weights):
    discrimination_scores = []
    for protected_instance in protected_indices_test:
        test_instance = test_set.iloc[protected_instance]

        if class_info_test.iloc[protected_instance] == 0:
            protected_neighbours, unprotected_neighbours = find_2k_nearest_neighbors_euclidean(k, test_instance,
                                                           training_set, protected_indices_train, unprotected_indices_train,
                                                           indices_info, weights)
            diff = calc_difference(protected_neighbours, unprotected_neighbours, class_info_train)
            discrimination_scores.append(diff)
        else:
            discrimination_scores.append(-1)
    return discrimination_scores


def make_distance_row_mahalanobis(training_set, instance, indices_info, weight_matrix):
    dist_row = cdist(instance.values[None,], training_set.values, mahalanobis_distance, weights=weight_matrix,
                     indices_info=indices_info)
    return dist_row[0]


def find_2k_nearest_neighbors_mahalanobis(k, instance, training_set, protected_indices, unprotected_indices, indices_info, weight_matrix):
    distance_row = pd.Series(make_distance_row_mahalanobis(training_set, instance, indices_info, weight_matrix))

    protected_instances = distance_row.iloc[protected_indices]
    unprotected_instances = distance_row.iloc[unprotected_indices]

    protected_neighbours_idx = np.argpartition(protected_instances, k)
    unprotected_neighbours_idx = np.argpartition(unprotected_instances, k)

    protected_neighbours = (protected_instances.iloc[protected_neighbours_idx[:k]])
    unprotected_neighbours = (unprotected_instances.iloc[unprotected_neighbours_idx[:k]])

    return protected_neighbours, unprotected_neighbours



def give_all_disc_scores_mahalanobis_2k_approach(k, class_info_train, protected_indices_train, unprotected_indices_train, training_set, protected_indices_test, class_info_test, test_set, indices_info, weight_matrix):
    discrimination_scores = []
    for protected_instance in protected_indices_test:
        test_instance = test_set.iloc[protected_instance]
        if class_info_test.iloc[protected_instance] == 0:
            protected_neighbours, unprotected_neighbours = find_2k_nearest_neighbors_mahalanobis(k, test_instance, training_set, protected_indices_train,
                                                                                        unprotected_indices_train, indices_info, weight_matrix)

            diff = calc_difference(protected_neighbours, unprotected_neighbours, class_info_train)
            discrimination_scores.append(diff)
        else:
            discrimination_scores.append(-1)
    return discrimination_scores



def give_Lenders_disc_score_with_reject_option(k, epsilon, class_info_train, unprotected_indices_train, training_set, protected_indices_test, class_info_test, test_set, indices_info, distance_function, weights):
    discrimination_scores = []
    rejected_indices = []
    train_set_unprotected = training_set.iloc[unprotected_indices_train].reset_index(drop=True)
    train_class_info_unprotected = class_info_train.iloc[unprotected_indices_train].reset_index(drop=True)
    for protected_instance in protected_indices_test:
        test_instance = test_set.iloc[protected_instance]
        if class_info_test.iloc[protected_instance] == 0:
            nearest_unprotected_neighbours_and_their_distances, distance_to_closest_neighbour = find_k_nearest_unprotected_neighbors(k, test_instance,
                                                                train_set_unprotected, indices_info, distance_function, weights)
            if (distance_to_closest_neighbour >= epsilon):
                discrimination_scores.append(-1000)
                rejected_indices.append(test_instance.name)
            else:
                class_nearest_neighbours = train_class_info_unprotected.iloc[nearest_unprotected_neighbours_and_their_distances.index]
                discrimination_score = sum(class_nearest_neighbours) / len(class_nearest_neighbours)
                discrimination_scores.append(discrimination_score)
        else:
            discrimination_scores.append(-1)
    return discrimination_scores, rejected_indices


def give_Lenders_disc_score_without_reject_option(k, class_info_train, unprotected_indices_train, training_set, protected_indices_test, class_info_test, test_set, indices_info, distance_function, weights):
    discrimination_scores = []
    train_set_unprotected = training_set.iloc[unprotected_indices_train].reset_index(drop=True)
    train_class_info_unprotected = class_info_train.iloc[unprotected_indices_train].reset_index(drop=True)
    distances_to_closest_neighbour = []
    for protected_instance in protected_indices_test:
        test_instance = test_set.iloc[protected_instance]
        if class_info_test.iloc[protected_instance] == 0:
            nearest_unprotected_neighbours_and_their_distances, distance_to_closest_neighbour = find_k_nearest_unprotected_neighbors(k, test_instance,
                                                                train_set_unprotected, indices_info, distance_function, weights)
            class_nearest_neighbours = train_class_info_unprotected.iloc[nearest_unprotected_neighbours_and_their_distances.index]
            discrimination_score = sum(class_nearest_neighbours) / len(class_nearest_neighbours)
            discrimination_scores.append(discrimination_score)
            distances_to_closest_neighbour.append(distance_to_closest_neighbour)
        else:
            discrimination_scores.append(-1)
    return discrimination_scores, distances_to_closest_neighbour

