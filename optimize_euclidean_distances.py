import numpy as np
from scipy.optimize import minimize, newton
import scipy.stats
import pandas as pd
import math
from optimize_distances_utils import calc_distances_within_and_between_classes, get_abs_difference_between_instances_with_same_and_different_class_label, \
    give_abs_difference_vector_between_instances


#This function gives the distance between two instances x and y as defined by Luong
def luong_distance(x, y, indices_info, weights=None):
    difference_vector = give_abs_difference_vector_between_instances(x, y, indices_info)
    return sum(difference_vector)/len(difference_vector)


#This function gives the weighted euclidean distance between two instances x and y
def weighted_euclidean_distance(x, y, weights, indices_info):
    abs_difference = give_abs_difference_vector_between_instances(x, y, indices_info)
    return euclidean_distance_given_abs_diff(abs_difference, weights)


def euclidean_distance_given_abs_diff(abs_diff, weights):
    sum_of_distances = 0
    for i in range(0, len(abs_diff)):
        sum_of_distances += weights[i] * (abs_diff[i]**2)
    if sum_of_distances == 0:
        return sum_of_distances
    return math.sqrt(sum_of_distances)


# def objective_weighted_euclidean(weights, protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l1_norm):
#     print(weights)
#     prot_dist_diff, prot_dist_same = calc_distances_within_and_between_classes(protected_labels, protected_data,
#                                                                                weights, indices_info, weighted_euclidean_distance)
#
#     unprot_dist_diff, unprot_dist_same = calc_distances_within_and_between_classes(unprotected_labels, unprotected_data,
#                                                                                    weights, indices_info, weighted_euclidean_distance)
#
#     dist_same_classes = prot_dist_same + unprot_dist_same
#     dist_diff_classes = prot_dist_diff + unprot_dist_diff
#
#     mean_dist_same = sum(dist_same_classes) / len(dist_same_classes)
#     mean_dist_diff = sum(dist_diff_classes) / len(dist_diff_classes)
#
#     l1_norm = lambda_l1_norm * sum(weights)
#     print(mean_dist_same - mean_dist_diff + l1_norm)
#
#     return mean_dist_same - mean_dist_diff + l1_norm

def objective_weighted_euclidean(weights, protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l1_norm):
    print(weights)
    prot_dist_diff, prot_dist_same = calc_distances_within_and_between_classes(protected_labels, protected_data,
                                                                               weights, indices_info, weighted_euclidean_distance)

    unprot_dist_diff, unprot_dist_same = calc_distances_within_and_between_classes(unprotected_labels, unprotected_data,
                                                                                   weights, indices_info, weighted_euclidean_distance)

    mean_prot_dist_diff = sum(prot_dist_diff) / len(prot_dist_diff)
    mean_prot_dist_same = sum(prot_dist_same) / len(prot_dist_same)

    mean_unprot_dist_diff = sum(unprot_dist_diff) / len(unprot_dist_diff)
    mean_unprot_dist_same = sum(unprot_dist_same) / len(unprot_dist_same)

    l1_norm = lambda_l1_norm * sum(weights)
    sum_of_mean_of_dist_diffs = mean_prot_dist_diff + mean_unprot_dist_diff
    sum_of_mean_of_dist_same = mean_prot_dist_same + mean_unprot_dist_same

    print(sum_of_mean_of_dist_same-sum_of_mean_of_dist_diffs+l1_norm)

    return sum_of_mean_of_dist_same - sum_of_mean_of_dist_diffs + l1_norm


def make_euclidean_derivative_per_label_group(number_of_attributes, label_group, weightarray):
    derivative_vector = []
    for i in range(number_of_attributes):
        sum_of_elements = 0
        for element in range(len(label_group)):
            euclidean_distance = euclidean_distance_given_abs_diff(label_group[element], weightarray)
            if euclidean_distance != 0:
                sum_of_elements += ((1/(2*euclidean_distance)) * (label_group[element][i] ** 2))
        derivative_vector.append(sum_of_elements)
    return derivative_vector

# def euclidean_derivative(weights, protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l1_norm):
#     prot_same, prot_diff = get_abs_difference_between_instances_with_same_and_different_class_label(protected_labels, protected_data,
#                                                                                                     indices_info)
#     unprot_same, unprot_diff = get_abs_difference_between_instances_with_same_and_different_class_label(unprotected_labels, unprotected_data,
#                                                                                                         indices_info)
#     number_of_same = len(prot_same) + len(unprot_same)
#     number_of_different = len(prot_diff) + len(unprot_diff)
#
#     derivative_prot_same = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), prot_same, weights))
#     derivative_prot_diff = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), prot_diff, weights))
#     derivative_unprot_same = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), unprot_same, weights))
#     derivative_unprot_diff = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), unprot_diff, weights))
#
#     derivative_same = 1 / number_of_same * (derivative_prot_same + derivative_unprot_same)
#     derivative_diff = 1 / number_of_different * (derivative_prot_diff + derivative_unprot_diff)
#
#     derivative = (derivative_same - derivative_diff + lambda_l1_norm)
#     print(derivative)
#     return derivative

def euclidean_derivative(weights, protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l1_norm):
    prot_same, prot_diff = get_abs_difference_between_instances_with_same_and_different_class_label(protected_labels, protected_data,
                                                                                                    indices_info)
    unprot_same, unprot_diff = get_abs_difference_between_instances_with_same_and_different_class_label(unprotected_labels, unprotected_data,
                                                                                                        indices_info)

    derivative_prot_same = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), prot_same, weights))
    derivative_prot_diff = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), prot_diff, weights))
    derivative_unprot_same = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), unprot_same, weights))
    derivative_unprot_diff = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), unprot_diff, weights))

    derivative_prot_same = 1/(len(prot_same)) * derivative_prot_same
    derivative_prot_diff = 1/(len(prot_diff)) * derivative_prot_diff
    derivative_unprot_same = 1/(len(unprot_same)) * derivative_unprot_same
    derivative_unprot_diff = 1/(len(unprot_diff)) * derivative_unprot_diff

    sum_derivative_same = derivative_prot_same + derivative_unprot_same
    sum_derivative_diff = derivative_prot_diff + derivative_unprot_diff

    derivative = (sum_derivative_same - sum_derivative_diff + lambda_l1_norm)
    print(derivative)
    return derivative


def optimize_weighted_euclidean(data, class_label, protected_attribute, indices_info, protected_label, unprotected_label, lambda_l1_norm):
    protected_labels = class_label[np.where(protected_attribute == protected_label)]
    unprotected_labels = class_label[np.where(protected_attribute == unprotected_label)]

    protected_data = data[np.where(protected_attribute == protected_label)]
    unprotected_data = data[np.where(protected_attribute == unprotected_label)]

    initial_weights = [0.1]*data.shape[1]

    b = (1e-11, float('inf'))
    bds = [b] * data.shape[1]

    # sol = minimize(objective_weighted_euclidean, initial_weights, (protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l1_norm),
    #                  method='L-BFGS-B', jac=euclidean_derivative, bounds=bds)
    # sol = newton(objective_weighted_euclidean, initial_weights, euclidean_derivative, args=(protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l1_norm))
    sol = minimize(objective_weighted_euclidean, initial_weights, (protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l1_norm),
                   method='SLSQP', jac=euclidean_derivative, bounds=bds)
    print(sol)
    return sol['x']





