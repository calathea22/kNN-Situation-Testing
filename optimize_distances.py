import numpy as np
from scipy.optimize import minimize
import scipy.stats
import pandas as pd
import math

#This function gives the distance between two instances x and y as defined by Luong
def luong_distance(x, y, indices_info):
    sum_of_distances = 0
    for i in range(0, len(x)):
        sum_of_distances += give_int_ordinal_or_nominal_difference(x[i], y[i], i, indices_info)
    return sum_of_distances/len(x)

#This function gives the weighted euclidean distance between two instances x and y
def weighted_euclidean_distance(x, y, weights, indices_info):
    sum_of_distances = 0
    for i in range(0, len(x)):
        sum_of_distances += weights[i] * give_int_ordinal_or_nominal_distance_euclidean(x[i], y[i], i, indices_info)

    return math.sqrt(sum_of_distances)

def euclidean_distance_given_abs_diff(abs_diff, weights):
    sum_of_distances = 0
    for i in range(0, len(abs_diff)):
        sum_of_distances += weights[i] * (abs_diff[i]**2)
    return math.sqrt(sum_of_distances)
    #return math.sqrt(sum_of_distances) / len(abs_diff)


def calc_distances_within_and_between_classes(labels, data, weights, indices_info, distance_function):
    dist_diff_classes = []
    dist_same_classes = []
    for i in range(0, len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                dist_diff_classes.append(distance_function(data[i], data[j], weights, indices_info))
            else:
                dist_same_classes.append(distance_function(data[i], data[j], weights, indices_info))
    return (dist_diff_classes, dist_same_classes)


def give_int_ordinal_or_nominal_difference(xi, yi, index, indices_info):
    interval_indices = indices_info['interval']
    ordinal_indices = indices_info['ordinal']

    if index in interval_indices or index in ordinal_indices:
        return abs(xi-yi)
    else:
        return xi != yi


def give_int_ordinal_or_nominal_distance_euclidean(xi, yi, index, indices_info):
    interval_indices = indices_info['interval']
    ordinal_indices = indices_info['ordinal']

    if index in interval_indices or index in ordinal_indices:
        return(xi-yi)**2
    else:
        return xi != yi

def give_difference_between_instances(x, y, indices_info):
    difference_vector = []
    for i in range(0, len(x)):
        difference_vector.append(give_int_ordinal_or_nominal_difference(x[i], y[i], i, indices_info))
    return np.array(difference_vector)


def get_instances_with_same_and_different_class_label(labels, data, indices_info):
    same_classes = []
    different_classes = []

    for i in range(0, len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                different_classes.append(give_difference_between_instances(data[i], data[j], indices_info))
            else:
                same_classes.append(give_difference_between_instances(data[i], data[j], indices_info))
    return(same_classes, different_classes)


def objective_weighted_euclidean(weights, protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l1_norm):
    prot_dist_diff, prot_dist_same = calc_distances_within_and_between_classes(protected_labels, protected_data,
                                                                               weights, indices_info, weighted_euclidean_distance)
    unprot_dist_diff, unprot_dist_same = calc_distances_within_and_between_classes(unprotected_labels, unprotected_data,
                                                                                   weights, indices_info, weighted_euclidean_distance)
    dist_same_classes = prot_dist_same + unprot_dist_same
    dist_diff_classes = prot_dist_diff + unprot_dist_diff

    mean_dist_same = sum(dist_same_classes) / len(dist_same_classes)
    mean_dist_diff = sum(dist_diff_classes) / len(dist_diff_classes)

    l1_norm = lambda_l1_norm * sum(weights)
    print(weights)
    print(mean_dist_same - mean_dist_diff + l1_norm)

    return mean_dist_same - mean_dist_diff + l1_norm


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

def euclidean_derivative(weights, protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l1_norm):
    prot_same, prot_diff = get_instances_with_same_and_different_class_label(protected_labels, protected_data,
                                                                             indices_info)
    unprot_same, unprot_diff = get_instances_with_same_and_different_class_label(unprotected_labels, unprotected_data,
                                                                              indices_info)
    number_of_same = len(prot_same) + len(unprot_same)
    number_of_different = len(prot_diff) + len(unprot_diff)

    derivative_prot_same = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), prot_same, weights))
    derivative_prot_diff = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), prot_diff, weights))
    derivative_unprot_same = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), unprot_same, weights))
    derivative_unprot_diff = np.array(make_euclidean_derivative_per_label_group(len(protected_data[0]), unprot_diff, weights))

    derivative_same = 1 / number_of_same * (derivative_prot_same + derivative_unprot_same)
    derivative_diff = 1 / number_of_different * (derivative_prot_diff + derivative_unprot_diff)

    derivative = (derivative_same - derivative_diff + lambda_l1_norm)
    return derivative


def optimize_weighted_euclidean(data, class_label, protected_attribute, indices_info, protected_label, unprotected_label):
    protected_labels = class_label[np.where(protected_attribute == protected_label)]
    unprotected_labels = class_label[np.where(protected_attribute == unprotected_label)]

    protected_data = data[np.where(protected_attribute == protected_label)]
    unprotected_data = data[np.where(protected_attribute == unprotected_label)]
    print(data.shape[1])
    initial_weights = [0.1]*data.shape[1]
    b = (0.0, float('inf'))
    bds = [b] * data.shape[1]

    lambda_l1_norm = 0.2

    sol = minimize(objective_weighted_euclidean, initial_weights, (protected_data, unprotected_data, protected_labels, unprotected_labels, indices_info, lambda_l1_norm),
                   method='SLSQP', jac=euclidean_derivative, bounds=bds)
    print(sol)
    return sol['x']

