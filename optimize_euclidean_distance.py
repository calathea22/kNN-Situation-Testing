import numpy as np
from scipy.optimize import minimize, newton
import scipy.stats
import pandas as pd
import math
from optimize_distances_utils import calc_distances_within_and_between_classes, get_abs_difference_between_instances_with_same_and_different_class_label, \
    give_abs_difference_vector_between_instances



#This function gives the weighted euclidean distance between two instances x and y
def squared_weighted_euclidean_distance(x, y, weights, indices_info):
    abs_difference = give_abs_difference_vector_between_instances(x, y, indices_info)
    return squared_euclidean_distance_given_abs_diff(abs_difference, weights)


def euclidean_distance_given_abs_diff(abs_diff, weights):
    sum_of_distances = 0
    for i in range(0, len(abs_diff)):
        sum_of_distances += weights[i] * (abs_diff[i]**2)
    if sum_of_distances == 0:
        return sum_of_distances
    return math.sqrt(sum_of_distances)


def squared_euclidean_distance_given_abs_diff(abs_diff, weights):
    sum_of_distances = 0
    for i in range(0, len(abs_diff)):
        sum_of_distances += weights[i] * (abs_diff[i]**2)
    return sum_of_distances


def objective_weighted_euclidean(weights, protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l2_norm):
    prot_dist_diff, prot_dist_same = calc_distances_within_and_between_classes(protected_labels, protected_data,
                                                                               weights, indices_info, squared_weighted_euclidean_distance)

    unprot_dist_diff, unprot_dist_same = calc_distances_within_and_between_classes(unprotected_labels, unprotected_data,
                                                                                   weights, indices_info, squared_weighted_euclidean_distance)

    mean_prot_dist_diff = sum(prot_dist_diff) / len(prot_dist_diff)
    mean_prot_dist_same = sum(prot_dist_same) / len(prot_dist_same)

    mean_unprot_dist_diff = sum(unprot_dist_diff) / len(unprot_dist_diff)
    mean_unprot_dist_same = sum(unprot_dist_same) / len(unprot_dist_same)

    #is changed to l1_norm in final version
    l2_norm = lambda_l2_norm * sum(weights ** 2)

    sum_of_mean_of_dist_diffs = mean_prot_dist_diff + mean_unprot_dist_diff
    sum_of_mean_of_dist_same = mean_prot_dist_same + mean_unprot_dist_same

    return sum_of_mean_of_dist_same - sum_of_mean_of_dist_diffs + l2_norm


def make_euclidean_derivative_per_label_group(number_of_attributes, label_group):
    derivative_vector = []
    for i in range(number_of_attributes):
        sum_of_elements = 0
        for element in range(len(label_group)):
            sum_of_elements += label_group[element][i] ** 2
        derivative_vector.append(sum_of_elements)
    return derivative_vector


def euclidean_derivative(weights, protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l2_norm):
    prot_same, prot_diff = get_abs_difference_between_instances_with_same_and_different_class_label(protected_labels, protected_data,
                                                                                                    indices_info)
    unprot_same, unprot_diff = get_abs_difference_between_instances_with_same_and_different_class_label(unprotected_labels, unprotected_data,
                                                                                                        indices_info)

    derivative_prot_same = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), prot_same))
    derivative_prot_diff = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), prot_diff))
    derivative_unprot_same = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), unprot_same))
    derivative_unprot_diff = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), unprot_diff))

    derivative_prot_same = 1/(len(prot_same)) * derivative_prot_same
    derivative_prot_diff = 1/(len(prot_diff)) * derivative_prot_diff
    derivative_unprot_same = 1/(len(unprot_same)) * derivative_unprot_same
    derivative_unprot_diff = 1/(len(unprot_diff)) * derivative_unprot_diff

    sum_derivative_same = derivative_prot_same + derivative_unprot_same
    sum_derivative_diff = derivative_prot_diff + derivative_unprot_diff

    derivative = (sum_derivative_same - sum_derivative_diff)

    #is changed to l1 norm in final version
    for i in range(len(weights)):
        derivative[i] += 2 * lambda_l2_norm * weights[i]

    return derivative


def optimize_weighted_euclidean(data, class_label, protected_attribute, indices_info, protected_label, unprotected_label, lambda_l2_norm):
    protected_labels = class_label[np.where(protected_attribute == protected_label)]
    unprotected_labels = class_label[np.where(protected_attribute == unprotected_label)]

    protected_data = data[np.where(protected_attribute == protected_label)]
    unprotected_data = data[np.where(protected_attribute == unprotected_label)]

    initial_weights = [0.1]*data.shape[1]

    b = (1e-14, float('inf'))
    bds = [b] * data.shape[1]

    sol = minimize(objective_weighted_euclidean, initial_weights, (protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l2_norm),
                   method='SLSQP', jac=euclidean_derivative, bounds=bds)
    print(sol)
    return sol['x']





