from load_data import load_data, load_optimization_info, split_test_sets
import numpy as np
from kNN_discrimination_discovery import give_decision_labels_unprotected_group
from perform_zhang_algorithm import get_zhang_decision_scores_unprotected_group
from optimize_euclidean_distances import luong_distance, weighted_euclidean_distance
from optimize_mahalanobis_distances import mahalanobis_distance
from scipy.spatial.distance import pdist, squareform, cdist
import matplotlib.pyplot as plt
from math import sqrt
from giving_discrimination_scores import give_disc_scores_one_technique


# this function is used for finding the best k bit
def decision_labels_properties_for_unprotected_group(location, indices_info, k, technique, unprotected_label, lambda_l1=0):
    loaded_train_data = load_data(location, "train")
    loaded_test_data = load_data(location, "val")

    train_data = loaded_train_data['data']
    train_data_standardized = loaded_train_data['standardized_data']
    train_protected_info = loaded_train_data['protected_info']
    train_class_label = loaded_train_data['class_label']
    train_unprotected_indices = list(np.where(train_protected_info == 2)[0])

    val_data = loaded_test_data['data']
    val_data_standardized = loaded_test_data['standardized_data']
    val_protected_info = loaded_test_data['protected_info']
    val_class_label = loaded_test_data['class_label']
    val_unprotected_indices = list(np.where(val_protected_info == unprotected_label)[0])

    if technique == 'baseline':
        predictions_negative_class, predictions_positive_class = give_decision_labels_unprotected_group(k, class_info_train=train_class_label,
                                                                   unprotected_indices_train=train_unprotected_indices,
                                                                   training_set=train_data,
                                                                   unprotected_indices_test=val_unprotected_indices,
                                                                   class_info_test=val_class_label,
                                                                   test_set=val_data,
                                                                   indices_info=indices_info,
                                                                   distance_function=luong_distance)
        return predictions_negative_class, predictions_positive_class
    elif technique == 'luong':
        predictions_negative_class, predictions_positive_class = give_decision_labels_unprotected_group(k, class_info_train=train_class_label,
                                                                   unprotected_indices_train=train_unprotected_indices,
                                                                   training_set=train_data_standardized,
                                                                   unprotected_indices_test=val_unprotected_indices,
                                                                   class_info_test=val_class_label,
                                                                   test_set=val_data_standardized,
                                                                   indices_info=indices_info,
                                                                   distance_function=luong_distance)
        return predictions_negative_class, predictions_positive_class
    elif technique == 'zhang':
        predictions_negative_class, predictions_positive_class = get_zhang_decision_scores_unprotected_group("adult", k=k, train_data=train_data,
                                                       train_sens_attribute=train_protected_info,
                                                       train_decision_attribute=train_class_label,
                                                       test_data=val_data,
                                                       test_sens_attribute=val_protected_info,
                                                       test_decision_attribute=val_class_label)
        print(predictions_positive_class)
        print(predictions_negative_class)
        return predictions_negative_class, predictions_positive_class

    if technique == 'euclidean':
        weights_euclidean = load_optimization_info(location, lambda_l1, 0.09)['weights_euclidean']
        predictions_negative_class, predictions_positive_class = give_decision_labels_unprotected_group(k, class_info_train=train_class_label,
                                                                   unprotected_indices_train=train_unprotected_indices,
                                                                   training_set=train_data_standardized,
                                                                   unprotected_indices_test=val_unprotected_indices,
                                                                   class_info_test=val_class_label,
                                                                   test_set=val_data_standardized,
                                                                   indices_info=indices_info,
                                                                   distance_function=weighted_euclidean_distance,
                                                                   weights=weights_euclidean)


        return predictions_negative_class, predictions_positive_class
    elif technique == 'mahalanobis':
        mahalanobis_matrix = load_optimization_info(location, lambda_l1, lambda_l1)['mahalanobis_matrix']
        predictions_negative_class, predictions_positive_class = give_decision_labels_unprotected_group(k, class_info_train=train_class_label,
                                                                   unprotected_indices_train=train_unprotected_indices,
                                                                   training_set=train_data_standardized,
                                                                   unprotected_indices_test=val_unprotected_indices,
                                                                   class_info_test=val_class_label,
                                                                   test_set=val_data_standardized,
                                                                   indices_info=indices_info,
                                                                   distance_function=mahalanobis_distance,
                                                                   weights=mahalanobis_matrix)
        return predictions_negative_class, predictions_positive_class
    return 0


# this function is solely to be used on weighted euclidean and mahalanobis approach, since this function aims to find the
# optimal k AND lambda
def find_best_k_and_lambda_based_on_unprotected_region(location, indices_info, possible_k_values, possible_lambda_values, technique):
    performance_dict = {}
    for lambda_l1 in possible_lambda_values:
        print("Lambda:" + str(lambda_l1))
        for k in possible_k_values:
            print("K:" + str(k))
            predictions_neg_class, predictions_pos_class = decision_labels_properties_for_unprotected_group(location, indices_info, k, technique, 2, lambda_l1)
            print(sum(predictions_neg_class)/len(predictions_neg_class))
            print(sum(predictions_pos_class)/len(predictions_pos_class))
            print(sum(predictions_neg_class)/len(predictions_neg_class) - sum(predictions_pos_class)/len(predictions_pos_class))
            performance_dict[(k, lambda_l1)] = (sum(predictions_neg_class)/len(predictions_neg_class)) - (sum(predictions_pos_class)/len(predictions_pos_class))
    print(performance_dict)
    best_k_best_lambda, best_sum_of_disc_scores = min(performance_dict.items(), key=lambda x:x[1])
    best_k = best_k_best_lambda[0]
    best_lambda = best_k_best_lambda[1]
    print(best_k)
    print(best_lambda)
    return best_lambda


def get_distances_to_k_nearest_neighbour(distance_matrix, k):
    nearest_neighbours_idx = np.argpartition(distance_matrix, k-1)
    k_th_nearest_neighbour_column_idx = nearest_neighbours_idx[:,k-1]
    distance_k_th_nearest_neighbours = np.take_along_axis(distance_matrix, k_th_nearest_neighbour_column_idx[:,None], axis=1)
    return distance_k_th_nearest_neighbours


def get_sorted_distances_to_k_neighbour(distance_matrix, k):
    nearest_neighbours_idx = np.argpartition(distance_matrix, k - 1)
    k_th_nearest_neighbour_column_idx = nearest_neighbours_idx[:, k - 1]
    distances_k_th_nearest_neighbours = np.take_along_axis(distance_matrix, k_th_nearest_neighbour_column_idx[:, None],
                                                          axis=1).flatten()
    print(distances_k_th_nearest_neighbours)
    sorted_distances = np.sort(distances_k_th_nearest_neighbours)
    print(sorted_distances)
    return sorted_distances


def sorted_k_distance_plot(k_distances, number_of_points, k, title):
    x_axis = np.arange(0, number_of_points)

    plt.plot(x_axis, k_distances, label=str(k))
    plt.legend()
    #plt.ylim((10**-8,10**1) )
    plt.yscale("log")
    plt.title(title)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.show()
    return


def find_knee_point_of_sorted_k_distance_plot_geometric(sorted_k_distances):
    max_index = 0
    max_distance = -1

    first_index = 0
    last_index = len(sorted_k_distances) - 1
    first_element = sorted_k_distances[first_index]
    last_element = sorted_k_distances[last_index]

    length_of_imaginary_line = sqrt((last_element-first_element)**2 + (last_index - first_index)**2)

    for i in range(0, len(sorted_k_distances)):
        current_element_of_array = sorted_k_distances[i]
        first_term = (last_element-first_element) * i
        second_term = (last_index-first_index) * current_element_of_array
        numerator = abs(first_term-second_term+(last_index*first_element) - (first_index*last_element))
        current_distance = numerator/length_of_imaginary_line
        print("i: " + str(i))
        print(current_distance)
        print("threshold: " + str(sorted_k_distances[i]))
        if current_distance > max_distance:
            max_distance = current_distance
            max_index = i
    print(max_index)
    print(sorted_k_distances[max_index])
    return sorted_k_distances[max_index]



def find_knee_point_of_sorted_k_distance_plot_largest_slope(sorted_k_distances):
    max_index = 0
    max_slope = 0
    for i in range(0, (len(sorted_k_distances)-1)):
        current_distance = sorted_k_distances[i]
        next_distance = sorted_k_distances[i+1]
        slope = next_distance - current_distance
        if (slope > max_slope):
            max_slope = slope
            max_index = i
    return sorted_k_distances[max_index]



def find_best_reject_threshold_based_on_k_distance_plot(location, indices_info, k, technique, lambda_l1, title):
    loaded_train_data = load_data(location, "train")
    loaded_test_data = load_data(location, "val")

    train_data_standardized = loaded_train_data['standardized_data']
    train_protected_info = loaded_train_data['protected_info']
    train_unprotected_indices = list(np.where(train_protected_info == 2)[0])

    val_data_standardized = loaded_test_data['standardized_data']
    val_class_label = loaded_train_data['class_label']
    val_protected_info = loaded_test_data['protected_info']

    val_indices_negative_class_label = set(np.where(val_class_label == 0)[0])
    val_protected_indices = set(np.where(val_protected_info == 1)[0])
    val_indices_protected_and_negative = val_indices_negative_class_label.intersection(val_protected_indices)

    train_data_standardized_unprotected = train_data_standardized.iloc[train_unprotected_indices]
    val_data_standardized_protected = val_data_standardized.iloc[list(val_indices_protected_and_negative)]

    loaded_optimization_info = load_optimization_info(location, lambda_l1, lambda_l1)

    if (technique == 'euclidean'):
        weights_euclidean = loaded_optimization_info['weights_euclidean']
        distance_matrix = cdist(val_data_standardized_protected.values, train_data_standardized_unprotected.values, weighted_euclidean_distance, weights= weights_euclidean, indices_info=indices_info)
    elif (technique == 'mahalanobis'):
        mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']
        distance_matrix = cdist(val_data_standardized_protected.values, train_data_standardized_unprotected.values, mahalanobis_distance, weights= mahalanobis_matrix, indices_info=indices_info)

    sorted_distances = get_sorted_distances_to_k_neighbour(distance_matrix, k)
    best_threshold = find_knee_point_of_sorted_k_distance_plot_geometric(sorted_distances)
    #best_threshold = find_knee_point_of_sorted_k_distance_plot_largest_slope(sorted_distances)
    print(best_threshold)
    sorted_k_distance_plot(sorted_distances, len(sorted_distances), k, title)

    return best_threshold



def find_best_reject_threshold_based_on_percentage_rejected(location, indices_info, k, technique, lambda_l1, desired_percent_rejected):
    loaded_train_data = load_data(location, "train")
    loaded_test_data = load_data(location, "val")
    loaded_optimization_info = load_optimization_info(location, lambda_l1, lambda_l1)

    disc_scores, dist_to_closest_neighbours = give_disc_scores_one_technique(loaded_train_data, loaded_test_data, loaded_optimization_info, technique, indices_info, k)
    dist_to_closest_neighbours = np.array(dist_to_closest_neighbours)

    print(disc_scores)
    print(len(disc_scores))
    print(len(dist_to_closest_neighbours))
    desired_amount_rejected = round(len(dist_to_closest_neighbours) * desired_percent_rejected)
    print(desired_amount_rejected)
    sorted_dist_to_closest_neighbours = -np.sort(-dist_to_closest_neighbours)
    print(sorted_dist_to_closest_neighbours)
    threshold = sorted_dist_to_closest_neighbours[desired_amount_rejected-1]
    return threshold


def find_best_k_based_on_unprotected_region(location, indices_info, possible_k_values, technique):
    performance_dict = {}
    for k in possible_k_values:
        print("K:" + str(k))
        predictions_neg_class, predictions_pos_class = decision_labels_properties_for_unprotected_group(location, indices_info, k, technique, 2)
        print(sum(predictions_neg_class)/len(predictions_neg_class))
        print(sum(predictions_pos_class)/len(predictions_pos_class))
        print(sum(predictions_neg_class)/len(predictions_neg_class) - sum(predictions_pos_class)/len(predictions_pos_class))
        performance_dict[k] = (sum(predictions_neg_class)/len(predictions_neg_class)) - (sum(predictions_pos_class)/len(predictions_pos_class))
        print(performance_dict)
    best_k, best_sum_of_disc_scores = min(performance_dict.items(), key=lambda x:x[1])
    print(best_k)
    return best_k


def find_best_threshold_helper_function(ordered_discrimination_scores, estimate_amount_of_discriminated_people):
    current_threshold = ordered_discrimination_scores[0]
    previous_threshold = ordered_discrimination_scores[0]
    previous_diff = 10000
    index = 0
    done = False

    while(not done):

        while(current_threshold == ordered_discrimination_scores[index]):
            index += 1
            if index==len(ordered_discrimination_scores):
                return previous_threshold

        current_diff = abs(estimate_amount_of_discriminated_people - index)

        if (current_diff == 0):
            done = True
        elif current_diff > previous_diff:
            current_threshold = previous_threshold
            done = True
        else:
            previous_threshold = current_threshold
            current_threshold = ordered_discrimination_scores[index]
            previous_diff = current_diff
    return current_threshold


def find_best_threshold_based_on_estimated_number_of_discriminated_people(location, indices_info, k, technique, estimated_percent_of_discrimination, lambda_l1=0):
    loaded_train_data = load_data(location, "train")
    loaded_val_data = load_data(location, "val")
    loaded_optimization_info = load_optimization_info(location, lambda_l1, lambda_l1)

    discrimination_scores = np.array(give_disc_scores_one_technique(loaded_train_data, loaded_val_data, loaded_optimization_info, technique, indices_info, k))
    print("Amount of women: " + str(len(discrimination_scores)))
    discrimination_scores_negative_class_labels_only = discrimination_scores[discrimination_scores != -1]
    amount_of_positive_class_labels = len(discrimination_scores) - len(discrimination_scores_negative_class_labels_only)
    print("Amount of positive class labels" + str(amount_of_positive_class_labels))
    print("Amount of negative class labels" + str(len(discrimination_scores_negative_class_labels_only)))
    estimate_amount_of_actual_positive_class_labels = int(1/(1-estimated_percent_of_discrimination) * amount_of_positive_class_labels)
    estimate_amount_of_discriminated_people = estimate_amount_of_actual_positive_class_labels - amount_of_positive_class_labels
    print("Estimated number of discriminated people:" + str(estimate_amount_of_discriminated_people))
    sorted_discrimination_scores_negative_class_labels = np.sort(discrimination_scores_negative_class_labels_only)
    reverse_sorted_disc_scores_neg_class_labels = sorted_discrimination_scores_negative_class_labels[::-1]
    best_threshold = find_best_threshold_helper_function(reverse_sorted_disc_scores_neg_class_labels, estimate_amount_of_discriminated_people)
    print(best_threshold)
    return best_threshold


def find_best_threshold_based_on_demographic_parity(location, indices_info, k, technique, lambda_l1=0):
    loaded_train_data = load_data(location, "train")
    loaded_val_data = load_data(location, "val")
    loaded_optimization_info = load_optimization_info(location, lambda_l1, lambda_l1)

    discrimination_scores = np.array(give_disc_scores_one_technique(loaded_train_data, loaded_val_data, loaded_optimization_info, technique, indices_info, k))

    discrimination_scores_negative_class_labels_only = discrimination_scores[discrimination_scores != -1]
    sorted_discrimination_scores_negative_class_labels = np.sort(discrimination_scores_negative_class_labels_only)
    reverse_sorted_disc_scores_neg_class_labels = sorted_discrimination_scores_negative_class_labels[::-1]

    loaded_test_data = load_data(location, "val")
    val_protected_info = loaded_test_data['protected_info']
    val_class_label = loaded_test_data['class_label']

    val_protected_indices = list(np.where(val_protected_info == 1)[0])
    val_unprotected_indices = list(np.where(val_protected_info == 2)[0])

    class_labels_unprotected = val_class_label[val_unprotected_indices]
    amount_of_positive_class_labels_unprotected = sum(class_labels_unprotected)
    print("Amount of positive class labels unprotected: " + str(amount_of_positive_class_labels_unprotected))

    class_labels_protected = val_class_label[val_protected_indices]
    amount_of_positive_class_labels_protected = sum(class_labels_protected)
    print("Amount of positive class labels protected: " + str(amount_of_positive_class_labels_protected))

    estimate_amount_of_discriminated_people = amount_of_positive_class_labels_unprotected - amount_of_positive_class_labels_protected
    print("Estimated amount of discriminated people: " + str(estimate_amount_of_discriminated_people))
    best_threshold = find_best_threshold_helper_function(reverse_sorted_disc_scores_neg_class_labels, estimate_amount_of_discriminated_people)
    print(best_threshold)
    return best_threshold

