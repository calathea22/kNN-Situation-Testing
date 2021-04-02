from load_data import load_data, load_optimization_info, split_test_sets
import numpy as np
from kNN_discrimination_discovery import give_decision_labels_unprotected_group
from perform_zhang_algorithm import get_zhang_decision_scores_unprotected_group
from optimize_distances_utils import luong_distance, weighted_euclidean_distance, mahalanobis_distance
import matplotlib.pyplot as plt
from giving_discrimination_scores import give_disc_scores_one_technique
from seaborn import boxplot, stripplot


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
        weights_euclidean = load_optimization_info(location, lambda_l1, lambda_l1)['weights_euclidean']
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


def find_best_threshold_based_on_unprotected_region(location, indices_info, k, technique, lambda_l1=0, adult_or_admission="admission"):
    loaded_train_data = load_data(location, "train")
    loaded_val_data = load_data(location, "val")
    loaded_optimization_info = load_optimization_info(location, lambda_l1, lambda_l1)

    disc_scores_protected = give_disc_scores_one_technique(loaded_train_data, loaded_val_data, loaded_optimization_info, technique, indices_info, k, 1, adult_or_admission)
    disc_scores_unprotected = give_disc_scores_one_technique(loaded_train_data, loaded_val_data, loaded_optimization_info, technique, indices_info, k, 2, adult_or_admission)

    disc_scores_protected = np.array(disc_scores_protected)
    disc_scores_unprotected = np.array(disc_scores_unprotected)

    disc_scores_protected_neg_class = disc_scores_protected[np.where(disc_scores_protected != -1)[0]]
    disc_scores_unprotected_neg_class = disc_scores_unprotected[np.where(disc_scores_unprotected != -1)[0]]

    first_quartile = np.quantile(disc_scores_unprotected_neg_class, 0.25)
    third_quartile = np.quantile(disc_scores_unprotected_neg_class, 0.75)
    inter_quartile_range = third_quartile - first_quartile
    max_non_outlier = third_quartile + (1.5 * inter_quartile_range)
    print(max_non_outlier)


    boxplot(data=[disc_scores_unprotected_neg_class, disc_scores_protected_neg_class], showmeans=True,
            meanprops={"marker": "o",
                       "markerfacecolor": "grey",
                       "markeredgecolor": "black",
                       "markersize": "8"})
    stripplot(data=[disc_scores_unprotected_neg_class, disc_scores_protected_neg_class], color=".3")

    plt.show()
    return max_non_outlier