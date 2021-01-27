from load_data import load_data, load_optimization_info, split_test_sets
import numpy as np
from kNN_discrimination_discovery import give_all_disc_scores_Luong, give_all_disc_scores_euclidean
import utils
from perform_zhang_algorithm import get_zhang_discrimination_scores

def area_under_curve_all_approaches(loaded_train_data, loaded_test_data, loaded_optimization_info, k_luong, k_zhang, k_euclidean):
    train_data = loaded_train_data['data']
    train_data_standardized = loaded_train_data['standardized_data']
    train_protected_info = loaded_train_data['protected_info']
    train_class_label = loaded_train_data['class_label']
    train_protected_indices = list(np.where(train_protected_info == 1)[0])
    train_unprotected_indices = list(np.where(train_protected_info == 2)[0])

    val_data = loaded_test_data['data']
    val_data_standardized = loaded_test_data['standardized_data']
    val_ground_truth = loaded_test_data['ground_truth']
    val_protected_info = loaded_test_data['protected_info']
    val_class_label = loaded_test_data['class_label']
    val_protected_indices = list(np.where(val_protected_info == 1)[0])

    indices_info = loaded_optimization_info['indices_info']
    weights_euclidean = loaded_optimization_info['weights_euclidean']

    luong_scores = give_all_disc_scores_Luong(k_luong, class_info_train=train_class_label,
                                              protected_indices_train=train_protected_indices,
                                              unprotected_indices_train=train_unprotected_indices,
                                              training_set=train_data_standardized,
                                              protected_indices_test=val_protected_indices,
                                              class_info_test=val_class_label, test_set=val_data_standardized,
                                              indices_info=indices_info)
    auc_luong = utils.get_auc_score(val_ground_truth, luong_scores)

    zhang_scores = get_zhang_discrimination_scores("admission", k=k_zhang, train_data=train_data,
                                                             train_sens_attribute=train_protected_info,
                                                             train_decision_attribute=train_class_label,
                                                             test_data=val_data,
                                                             test_sens_attribute=val_protected_info,
                                                             test_decision_attribute=val_class_label)
    auc_zhang = utils.get_auc_score(val_ground_truth, zhang_scores)


    weighted_euclidean_scores = give_all_disc_scores_euclidean(k_euclidean, class_info_train=train_class_label,
                                                               protected_indices_train=train_protected_indices,
                                                               unprotected_indices_train=train_unprotected_indices,
                                                               training_set=train_data_standardized,
                                                               protected_indices_test=val_protected_indices,
                                                               class_info_test=val_class_label,
                                                               test_set=val_data_standardized,
                                                               indices_info=indices_info,
                                                               weights=weights_euclidean)
    auc_euclidean = utils.get_auc_score(val_ground_truth, weighted_euclidean_scores)

    return auc_luong, auc_zhang, auc_euclidean


def area_under_curve_validation_set(location, k_array):
    loaded_train_data = load_data(location, "train")
    loaded_val_data = load_data(location, "val")
    loaded_optimization_info = load_optimization_info(location)

    for k in k_array:
        print(k)
        print(area_under_curve_all_approaches(loaded_train_data, loaded_val_data, loaded_optimization_info, k,
                                        k, k))
    return



def area_under_curve_test_sets(location, n_splits, k_luong, k_zhang, k_euclidean):
    loaded_train_data = load_data(location, "train")
    loaded_optimization_info = load_optimization_info(location)

    splitted_test_sets = split_test_sets(n_splits, location)
    i = 0
    results = []
    for test_set in splitted_test_sets:
        print("Split: " + str(i))
        auc_luong, auc_zhang, auc_euclidean = area_under_curve_all_approaches(
            loaded_train_data, test_set, loaded_optimization_info, k_luong, k_zhang,
            k_euclidean)

        results.append({'luong': auc_luong, 'zhang': auc_zhang, 'euclidean': auc_euclidean})
        print(results)
        i += 1
    # baseline_results = [result['baseline'] for result in results]
    # print(sum(baseline_results) / len(baseline_results))
    luong_results = [result['luong'] for result in results]
    print(sum(luong_results) / len(luong_results))
    zhang_results = [result['zhang'] for result in results]
    print(sum(zhang_results) / len(zhang_results))
    euclidean_results = [result['euclidean'] for result in results]
    print(sum(euclidean_results) / len(euclidean_results))
    # mahalanobis_results = [result['mahalanobis'] for result in results]
    # print(sum(mahalanobis_results) / len(mahalanobis_results))
    return results
