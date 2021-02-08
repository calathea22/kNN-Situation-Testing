from load_data import load_data, load_optimization_info, split_test_sets
import numpy as np
from kNN_discrimination_discovery import give_all_disc_scores_Luong_unprotected_group, \
    give_all_disc_scores_euclidean_unprotected_group, give_all_disc_scores_mahalanobis_unprotected_group
from perform_zhang_algorithm import get_zhang_discrimination_scores


# this function is used for finding the best k bit
def give_discrimination_scores_for_unprotected_group(location, k, technique, unprotected_label, lambda_l1=0):
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

    indices_info = {'interval': [0, 1, 2], 'ordinal': [], 'nominal': []}

    if technique == 'baseline':
        baseline_scores = give_all_disc_scores_Luong_unprotected_group(k, class_info_train=train_class_label,
                                                     unprotected_indices_train=train_unprotected_indices,
                                                     training_set=train_data, unprotected_indices_test=val_unprotected_indices,
                                                     class_info_test=val_class_label, test_set=val_data,
                                                     indices_info=indices_info)
        return baseline_scores
    elif technique == 'luong':
        luong_scores = give_all_disc_scores_Luong_unprotected_group(k, class_info_train=train_class_label,
                                              unprotected_indices_train=train_unprotected_indices,
                                              training_set=train_data_standardized,
                                              unprotected_indices_test=val_unprotected_indices,
                                              class_info_test=val_class_label, test_set=val_data_standardized,
                                              indices_info=indices_info)
        return luong_scores
    elif technique == 'zhang':
        zhang_scores = get_zhang_discrimination_scores("admission", k=k, train_data=train_data,
                                                       train_sens_attribute=train_protected_info,
                                                       train_decision_attribute=train_class_label,
                                                       test_data=val_data,
                                                       test_sens_attribute=val_protected_info,
                                                       test_decision_attribute=val_class_label)
        return zhang_scores
    elif technique == 'euclidean':
        weights_euclidean = load_optimization_info(location, lambda_l1, 0.2)['weights_euclidean']
        weighted_euclidean_scores = give_all_disc_scores_euclidean_unprotected_group(k, class_info_train=train_class_label,
                                                                   unprotected_indices_train=train_unprotected_indices,
                                                                   training_set=train_data_standardized,
                                                                   unprotected_indices_test=val_unprotected_indices,
                                                                   class_info_test=val_class_label,
                                                                   test_set=val_data_standardized,
                                                                   indices_info=indices_info,
                                                                   weights=weights_euclidean)
        return weighted_euclidean_scores
    elif technique == 'mahalanobis':
        mahalanobis_matrix = load_optimization_info(location, 0.2, lambda_l1)['mahalanobis_matrix']
        mahalanobis_scores = give_all_disc_scores_mahalanobis_unprotected_group(k, class_info_train=train_class_label,
                                                                   unprotected_indices_train=train_unprotected_indices,
                                                                   training_set=train_data_standardized,
                                                                   unprotected_indices_test=val_unprotected_indices,
                                                                   class_info_test=val_class_label,
                                                                   test_set=val_data_standardized,
                                                                   indices_info=indices_info,
                                                                   weight_matrix=mahalanobis_matrix)
        return mahalanobis_scores
    else:
        return 0

# this function is solely to be used on weighted euclidean and mahalanobis approach, since this function aims to find the
# optimal k AND lambda
def find_best_k_and_lambda_based_on_unprotected_region(location, possible_k_values, possible_lambda_values, technique):
    performance_dict = {}
    for lambda_l1 in possible_lambda_values:
        print("Lambda:" + str(lambda_l1))
        for k in possible_k_values:
            print("K:" + str(k))
            discrimination_scores = np.array(give_discrimination_scores_for_unprotected_group(location, k, technique, 2, lambda_l1))
            discrimination_scores_negative_class_labels_only = discrimination_scores[discrimination_scores!=-1]
            performance_dict[(k, lambda_l1)] = sum(discrimination_scores_negative_class_labels_only)
            print(sum(discrimination_scores_negative_class_labels_only))
    target = 0.00
    print(performance_dict)
    best_k_best_lambda, best_sum_of_disc_scores = min(performance_dict.items(), key=lambda key_value_pair: abs(key_value_pair[1] - target))
    best_k = best_k_best_lambda[0]
    best_lambda = best_k_best_lambda[1]
    print(best_k)
    print(best_lambda)
    return best_k

#to be used for other approaches
def find_best_k_based_on_unprotected_region(location, possible_k_values, technique):
    k_performance_dict = {}
    for k in possible_k_values:
        discrimination_scores = np.array(give_discrimination_scores_for_unprotected_group(location, k, technique, 2, lambda_l1))
        discrimination_scores_negative_class_labels_only = discrimination_scores[discrimination_scores!=-1]
        k_performance_dict[k] = sum(discrimination_scores_negative_class_labels_only)

    target = 0.00
    print(k_performance_dict)
    best_k, best_sum_of_disc_scores = min(k_performance_dict.items(), key=lambda key_value_pair: abs(key_value_pair[1] - target))
    print(best_k)
    return best_k