from load_data import load_data, load_optimization_info, split_test_sets
import numpy as np
from kNN_discrimination_discovery import give_all_disc_scores_Luong, give_all_disc_scores_euclidean, give_all_disc_scores_Luong_unprotected_group, \
    give_all_disc_scores_euclidean_unprotected_group, give_all_disc_scores_mahalanobis_unprotected_group, give_all_disc_scores_mahalanobis
import utils
from perform_zhang_algorithm import get_zhang_discrimination_scores
import operator
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score


def area_under_curve_all_approaches(loaded_train_data, loaded_test_data, loaded_optimization_info, k_info):
    train_data = loaded_train_data['data']
    train_data_standardized = loaded_train_data['standardized_data']
    train_protected_info = loaded_train_data['protected_info']
    train_class_label = loaded_train_data['class_label']
    train_protected_indices = list(np.where(train_protected_info == 1)[0])
    train_unprotected_indices = list(np.where(train_protected_info == 2)[0])

    indices_info = {'interval': [0, 1, 2], 'ordinal': [], 'nominal': []}

    val_data = loaded_test_data['data']
    val_data_standardized = loaded_test_data['standardized_data']
    val_ground_truth = loaded_test_data['ground_truth']
    val_protected_info = loaded_test_data['protected_info']
    val_class_label = loaded_test_data['class_label']
    val_protected_indices = list(np.where(val_protected_info == 1)[0])

    weights_euclidean = loaded_optimization_info['weights_euclidean']
    mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']

    # baseline_scores = give_all_disc_scores_Luong(k_info['baseline'], class_info_train=train_class_label,
    #                                           protected_indices_train=train_protected_indices,
    #                                           unprotected_indices_train=train_unprotected_indices,
    #                                           training_set=train_data,
    #                                           protected_indices_test=val_protected_indices,
    #                                           class_info_test=val_class_label, test_set=val_data_standardized,
    #                                           indices_info=indices_info)
    # auc_baseline = utils.get_precision_recall_auc_score(val_ground_truth, baseline_scores)
    # print(auc_baseline)
    #
    # luong_scores = give_all_disc_scores_Luong(k_info['luong'], class_info_train=train_class_label,
    #                                           protected_indices_train=train_protected_indices,
    #                                           unprotected_indices_train=train_unprotected_indices,
    #                                           training_set=train_data_standardized,
    #                                           protected_indices_test=val_protected_indices,
    #                                           class_info_test=val_class_label, test_set=val_data_standardized,
    #                                           indices_info=indices_info)
    # auc_luong = utils.get_precision_recall_auc_score(val_ground_truth, luong_scores)
    # print(auc_luong)
    #
    #
    # zhang_scores = get_zhang_discrimination_scores("admission", k=k_info['zhang'], train_data=train_data,
    #                                                          train_sens_attribute=train_protected_info,
    #                                                          train_decision_attribute=train_class_label,
    #                                                          test_data=val_data,
    #                                                          test_sens_attribute=val_protected_info,
    #                                                          test_decision_attribute=val_class_label)
    # auc_zhang = utils.get_precision_recall_auc_score(val_ground_truth, zhang_scores)
    # print(auc_zhang)
    #
    weighted_euclidean_scores = give_all_disc_scores_euclidean(k_info['euclidean'], class_info_train=train_class_label,
                                                               protected_indices_train=train_protected_indices,
                                                               unprotected_indices_train=train_unprotected_indices,
                                                               training_set=train_data_standardized,
                                                               protected_indices_test=val_protected_indices,
                                                               class_info_test=val_class_label,
                                                               test_set=val_data_standardized,
                                                               indices_info=indices_info,
                                                               weights=weights_euclidean)
    auc_euclidean = utils.get_precision_recall_auc_score(val_ground_truth, weighted_euclidean_scores)
    print(auc_euclidean)

    mahalanobis_scores = give_all_disc_scores_mahalanobis(k_info['mahalanobis'], class_info_train=train_class_label,
                                                               protected_indices_train=train_protected_indices,
                                                               unprotected_indices_train=train_unprotected_indices,
                                                               training_set=train_data_standardized,
                                                               protected_indices_test=val_protected_indices,
                                                               class_info_test=val_class_label,
                                                               test_set=val_data_standardized,
                                                               indices_info=indices_info,
                                                               weight_matrix=mahalanobis_matrix)
    auc_mahalanobis = utils.get_precision_recall_auc_score(val_ground_truth, mahalanobis_scores)
    print(auc_mahalanobis)

    return auc_baseline, auc_luong, auc_zhang, auc_euclidean, auc_mahalanobis


def f1_score_all_approaches(loaded_train_data, loaded_test_data, loaded_optimization_info, k_info, threshold_info):
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

    baseline_scores = give_all_disc_scores_Luong(k_info['baseline'], class_info_train=train_class_label,
                                                 protected_indices_train=train_protected_indices,
                                                 unprotected_indices_train=train_unprotected_indices,
                                                 training_set=train_data, protected_indices_test=val_protected_indices,
                                                 class_info_test=val_class_label, test_set=val_data,
                                                 indices_info=indices_info)

    luong_scores = give_all_disc_scores_Luong(k_info['luong'], class_info_train=train_class_label,
                                              protected_indices_train=train_protected_indices,
                                              unprotected_indices_train=train_unprotected_indices,
                                              training_set=train_data_standardized,
                                              protected_indices_test=val_protected_indices,
                                              class_info_test=val_class_label, test_set=val_data_standardized,
                                              indices_info=indices_info)

    zhang_scores = get_zhang_discrimination_scores("admission", k=k_info['zhang'], train_data=train_data,
                                                             train_sens_attribute=train_protected_info,
                                                             train_decision_attribute=train_class_label,
                                                             test_data=val_data,
                                                             test_sens_attribute=val_protected_info,
                                                             test_decision_attribute=val_class_label)

    weighted_euclidean_scores= give_all_disc_scores_euclidean(k_info['euclidean'], class_info_train=train_class_label,
                                                               protected_indices_train=train_protected_indices,
                                                               unprotected_indices_train=train_unprotected_indices,
                                                               training_set=train_data_standardized,
                                                               protected_indices_test=val_protected_indices,
                                                               class_info_test=val_class_label,
                                                               test_set=val_data_standardized,
                                                               indices_info=indices_info,
                                                               weights=weights_euclidean)


    disc_labels_baseline = utils.give_disc_label(baseline_scores, threshold_info['baseline'])
    disc_labels_luong = utils.give_disc_label(luong_scores, threshold_info['luong'])
    disc_labels_zhang = utils.give_disc_label(zhang_scores, threshold_info['zhang'])
    disc_labels_euclidean = utils.give_disc_label(weighted_euclidean_scores, threshold_info['euclidean'])

    precision_baseline = precision_score(val_ground_truth, disc_labels_baseline)
    recall_baseline = recall_score(val_ground_truth, disc_labels_baseline)
    f1_score_baseline = f1_score(val_ground_truth, disc_labels_baseline)

    precision_luong = precision_score(val_ground_truth, disc_labels_luong)
    recall_luong = recall_score(val_ground_truth, disc_labels_luong)
    f1_score_luong = f1_score(val_ground_truth, disc_labels_luong)

    precision_zhang = precision_score(val_ground_truth, disc_labels_zhang)
    recall_zhang = recall_score(val_ground_truth, disc_labels_zhang)
    f1_score_zhang = f1_score(val_ground_truth, disc_labels_zhang)

    precision_euclidean = precision_score(val_ground_truth, disc_labels_euclidean)
    recall_euclidean = recall_score(val_ground_truth, disc_labels_euclidean)
    f1_score_euclidean = f1_score(val_ground_truth, disc_labels_euclidean)

    return f1_score_baseline, f1_score_luong, f1_score_zhang, f1_score_euclidean, \
            precision_baseline, precision_luong, precision_zhang, precision_euclidean, \
            recall_baseline, recall_luong, recall_zhang, recall_euclidean


def area_under_curve_validation_set(location, k_info, best_lambda_euclidean, best_lambda_mahalanobis):
    loaded_train_data = load_data(location, "train")
    loaded_val_data = load_data(location, "val")
    loaded_optimization_info = load_optimization_info(location, best_lambda_euclidean, best_lambda_mahalanobis)

    area_under_curve_all_approaches(loaded_train_data, loaded_val_data, loaded_optimization_info, k_info)
    return


def find_best_threshold_and_k_based_on_val_F1(location, possible_k_values, possible_threshold_values):
    loaded_train_data = load_data(location, "train")
    loaded_val_data = load_data(location, "val")
    indices_info = {'interval': [0, 1, 2], 'ordinal': [], 'nominal': []}
    loaded_optimization_info = load_optimization_info(location)

    train_data = loaded_train_data['data']
    train_data_standardized = loaded_train_data['standardized_data']
    train_protected_info = loaded_train_data['protected_info']
    train_class_label = loaded_train_data['class_label']
    train_protected_indices = list(np.where(train_protected_info == 1)[0])
    train_unprotected_indices = list(np.where(train_protected_info == 2)[0])

    val_data = loaded_val_data['data']
    val_data_standardized = loaded_val_data['standardized_data']
    val_ground_truth = loaded_val_data['ground_truth']
    val_protected_info = loaded_val_data['protected_info']
    val_class_label = loaded_val_data['class_label']
    val_protected_indices = list(np.where(val_protected_info == 1)[0])

    indices_info = loaded_optimization_info['indices_info']
    weights_euclidean = loaded_optimization_info['weights_euclidean']

    f1_scores = {}

    for k in possible_k_values:
        discrimination_scores = get_zhang_discrimination_scores("admission", k=k, train_data=train_data,
                                                             train_sens_attribute=train_protected_info,
                                                             train_decision_attribute=train_class_label,
                                                             test_data=val_data,
                                                             test_sens_attribute=val_protected_info,
                                                             test_decision_attribute=val_class_label)
        for t in possible_threshold_values:
            disc_labels = utils.give_disc_label(discrimination_scores, t)
            f1_score_disc_detection = f1_score(val_ground_truth, disc_labels)
            f1_scores[(k, t)] = f1_score_disc_detection
            print(k)
            print(t)
            print(f1_score_disc_detection)

    position_best_f1_score = max(f1_scores.items(), key=operator.itemgetter(1))[0]
    best_k = position_best_f1_score[0]
    best_threshold = position_best_f1_score[1]
    best_f1_score = f1_scores[position_best_f1_score]
    print('Maximum f1 score: %.4f' % (best_f1_score))
    print('Associated k: ' + str(best_k))
    print('Associated threshold: ' + str(best_threshold))

    return best_k, best_threshold


def area_under_curve_test_sets(location, n_splits, k_info, best_lambda_euclidean, best_lambda_mahalanobis):
    loaded_train_data = load_data(location, "train")
    loaded_optimization_info = load_optimization_info(location, best_lambda_euclidean, best_lambda_mahalanobis)

    splitted_test_sets = split_test_sets(n_splits, location)
    i = 0
    results = []
    for test_set in splitted_test_sets:
        print("Split: " + str(i))
        auc_baseline, auc_luong, auc_zhang, auc_euclidean, auc_mahalanobis = area_under_curve_all_approaches(
            loaded_train_data, test_set, loaded_optimization_info, k_info)
        results.append({'baseline': auc_baseline, 'luong': auc_luong, 'zhang': auc_zhang, 'euclidean': auc_euclidean})
        print(results)
        i += 1
    baseline_results = [result['baseline'] for result in results]
    print(sum(baseline_results) / len(baseline_results))
    luong_results = [result['luong'] for result in results]
    print(sum(luong_results) / len(luong_results))
    zhang_results = [result['zhang'] for result in results]
    print(sum(zhang_results) / len(zhang_results))
    euclidean_results = [result['euclidean'] for result in results]
    print(sum(euclidean_results) / len(euclidean_results))
    # mahalanobis_results = [result['mahalanobis'] for result in results]
    # print(sum(mahalanobis_results) / len(mahalanobis_results))
    return results


def f1_score_test_sets(location, n_splits, k_info, threshold_info):
    loaded_train_data = load_data(location, "train")
    loaded_optimization_info = load_optimization_info(location)

    splitted_test_sets = split_test_sets(n_splits, location)
    i = 0
    f1_results = []
    precision_results = []
    recall_results = []
    for test_set in splitted_test_sets:
        print("Split: " + str(i))
        f1_baseline, f1_luong, f1_zhang, f1_euclidean, \
        precision_baseline, precision_luong, precision_zhang, precision_euclidean, \
        recall_baseline, recall_luong, recall_zhang, recall_euclidean = f1_score_all_approaches(
            loaded_train_data, test_set, loaded_optimization_info, k_info, threshold_info)
        f1_results.append({'baseline': f1_baseline, 'luong': f1_luong, 'zhang': f1_zhang, 'euclidean': f1_euclidean})
        precision_results.append({'baseline': precision_baseline, 'luong': precision_luong, 'zhang': precision_zhang, 'euclidean': precision_euclidean})
        recall_results.append({'baseline': recall_baseline, 'luong': recall_luong, 'zhang': recall_zhang, 'euclidean': recall_euclidean})
        i += 1
    print("F1")
    utils.print_avg_results_from_dictionary(f1_results)
    print("Precision")
    utils.print_avg_results_from_dictionary(precision_results)
    print("Recall")
    utils.print_avg_results_from_dictionary(recall_results)


