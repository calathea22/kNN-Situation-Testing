import numpy as np
from baseline_disc_detection import give_disc_scores_baseline
from kNN_discrimination_discovery import give_all_disc_scores_Luong, give_Lenders_disc_score_without_reject_option, \
    weighted_euclidean_distance, mahalanobis_distance, luong_distance, give_Lenders_disc_score_with_reject_option, give_all_disc_scores_euclidean_2k_approach, \
    give_all_disc_scores_mahalanobis_2k_approach
from perform_zhang_algorithm import get_zhang_discrimination_scores


def give_all_disc_scores(loaded_train_data, loaded_test_data, loaded_optimization_info, indices_info, k_info, adult_or_admission):
    train_data = loaded_train_data['data']
    train_data_standardized = loaded_train_data['standardized_data']
    train_protected_info = loaded_train_data['protected_info']
    train_class_label = loaded_train_data['class_label']
    train_protected_indices = list(np.where(train_protected_info == 1)[0])
    train_unprotected_indices = list(np.where(train_protected_info == 2)[0])

    val_data = loaded_test_data['data']
    val_data_standardized = loaded_test_data['standardized_data']
    val_protected_info = loaded_test_data['protected_info']
    val_class_label = loaded_test_data['class_label']
    val_protected_indices = list(np.where(val_protected_info == 1)[0])
    val_ground_truth = loaded_test_data['ground_truth']

    weights_euclidean = loaded_optimization_info['weights_euclidean']
    mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']

    #
    baseline_scores = give_disc_scores_baseline(adult_or_admission, class_info_train=train_class_label, unprotected_indices_train=train_unprotected_indices,
                                                training_set=train_data, class_info_test=val_class_label,
                                                protected_indices_test=val_protected_indices, test_set=val_data)
    print(baseline_scores)
    # luong_scores = give_all_disc_scores_Luong(k_info['luong'], class_info_train=train_class_label,
    #                                           protected_indices_train=train_protected_indices,
    #                                           unprotected_indices_train=train_unprotected_indices,
    #                                           training_set=train_data_standardized,
    #                                           protected_indices_test=val_protected_indices,
    #                                           class_info_test=val_class_label, test_set=val_data_standardized,
    #                                           indices_info=indices_info)
    luong_scores_1k, _ = give_Lenders_disc_score_without_reject_option(k=k_info['luong'], class_info_train=train_class_label,
                         unprotected_indices_train=train_unprotected_indices, training_set=train_data_standardized, protected_indices_test=val_protected_indices,
                         class_info_test=val_class_label, test_set=val_data_standardized, indices_info=indices_info, distance_function=luong_distance,
                         weights=[])
    print(luong_scores_1k)
    # #
    # #
    zhang_scores = get_zhang_discrimination_scores(adult_or_admission, k=k_info['zhang'], train_data=train_data,
                                                             train_sens_attribute=train_protected_info,
                                                             train_decision_attribute=train_class_label,
                                                             test_data=val_data,
                                                             test_sens_attribute=val_protected_info,
                                                             test_decision_attribute=val_class_label)

    weighted_euclidean_scores, _ = give_Lenders_disc_score_without_reject_option(k=k_info['euclidean'], class_info_train=train_class_label,
                         unprotected_indices_train=train_unprotected_indices, training_set=train_data_standardized, protected_indices_test=val_protected_indices,
                         class_info_test=val_class_label, test_set=val_data_standardized, indices_info=indices_info, distance_function=weighted_euclidean_distance,
                         weights=weights_euclidean)
    print(weighted_euclidean_scores)
    #
    # weighted_euclidean_scores_2k = give_all_disc_scores_euclidean_2k_approach(k=k_info['euclidean'], class_info_train=train_class_label, protected_indices_train= train_protected_indices,
    #                      unprotected_indices_train=train_unprotected_indices, training_set=train_data_standardized, protected_indices_test=val_protected_indices,
    #                      class_info_test=val_class_label, test_set=val_data_standardized, indices_info=indices_info, weights=weights_euclidean)

    mahalanobis_scores, _ = give_Lenders_disc_score_without_reject_option(k=k_info['mahalanobis'], class_info_train=train_class_label,
                         unprotected_indices_train=train_unprotected_indices, training_set=train_data_standardized, protected_indices_test=val_protected_indices,
                         class_info_test=val_class_label, test_set=val_data_standardized, indices_info=indices_info, distance_function=mahalanobis_distance,
                         weights=mahalanobis_matrix)
    print(mahalanobis_scores)
    #
    # mahalanobis_scores_2k = give_all_disc_scores_mahalanobis_2k_approach(k=k_info['mahalanobis'], class_info_train=train_class_label, protected_indices_train= train_protected_indices,
    #                      unprotected_indices_train=train_unprotected_indices, training_set=train_data_standardized, protected_indices_test=val_protected_indices,
    #                      class_info_test=val_class_label, test_set=val_data_standardized, indices_info=indices_info, weight_matrix=mahalanobis_matrix)
    # baseline_scores = np.zeros(len(val_ground_truth))
    # luong_scores_1k = np.zeros(len(val_ground_truth))
    # zhang_scores = np.zeros(len(val_ground_truth))
    #mahalanobis_scores_2k = np.zeros(len(val_ground_truth))

    return baseline_scores, luong_scores_1k, zhang_scores, weighted_euclidean_scores, mahalanobis_scores


def give_disc_scores_one_technique(loaded_train_data, loaded_test_data, loaded_optimization_info, technique, indices_info, k):
    train_data = loaded_train_data['data']
    train_data_standardized = loaded_train_data['standardized_data']
    train_protected_info = loaded_train_data['protected_info']
    train_class_label = loaded_train_data['class_label']
    train_protected_indices = list(np.where(train_protected_info == 1)[0])
    train_unprotected_indices = list(np.where(train_protected_info == 2)[0])

    val_data = loaded_test_data['data']
    val_data_standardized = loaded_test_data['standardized_data']
    val_protected_info = loaded_test_data['protected_info']
    val_class_label = loaded_test_data['class_label']
    val_protected_indices = list(np.where(val_protected_info == 1)[0])

    if technique == 'baseline':
        baseline_scores = give_disc_scores_baseline(class_info_train=train_class_label,
                                                    unprotected_indices_train=train_unprotected_indices,
                                                    training_set=train_data, class_info_test=val_class_label,
                                                    protected_indices_test=val_protected_indices, test_set=val_data)
        return baseline_scores
    elif technique == 'luong':
        luong_scores = give_all_disc_scores_Luong(k, class_info_train=train_class_label,
                                                  protected_indices_train=train_protected_indices,
                                                  unprotected_indices_train=train_unprotected_indices,
                                                  training_set=train_data_standardized,
                                                  protected_indices_test=val_protected_indices,
                                                  class_info_test=val_class_label,
                                                  test_set=val_data_standardized, indices_info=indices_info)
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
        weights_euclidean = loaded_optimization_info['weights_euclidean']
        weighted_euclidean_scores, dist_to_closest_neighbours = give_Lenders_disc_score_without_reject_option(k, class_info_train=train_class_label,
                                                                                  unprotected_indices_train=train_unprotected_indices,
                                                                                  training_set=train_data_standardized,
                                                                                  protected_indices_test=val_protected_indices,
                                                                                  class_info_test=val_class_label,
                                                                                  test_set=val_data_standardized,
                                                                                  indices_info=indices_info,
                                                                                  distance_function=weighted_euclidean_distance,
                                                                                  weights=weights_euclidean)
        return weighted_euclidean_scores, dist_to_closest_neighbours
    elif technique == 'mahalanobis':
        mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']
        mahalanobis_scores, dist_to_closest_neighbours = give_Lenders_disc_score_without_reject_option(k, class_info_train=train_class_label,
                                                                           unprotected_indices_train=train_unprotected_indices,
                                                                           training_set=train_data_standardized,
                                                                           protected_indices_test=val_protected_indices,
                                                                           class_info_test=val_class_label,
                                                                           test_set=val_data_standardized,
                                                                           indices_info=indices_info,
                                                                           distance_function=mahalanobis_distance,
                                                                           weights=mahalanobis_matrix)
        return mahalanobis_scores, dist_to_closest_neighbours

    return 0


def give_disc_scores_with_reject_one_technique(loaded_train_data, loaded_test_data, loaded_optimization_info, technique, indices_info, k, reject_threshold):
    train_data_standardized = loaded_train_data['standardized_data']
    train_protected_info = loaded_train_data['protected_info']
    train_class_label = loaded_train_data['class_label']
    train_unprotected_indices = list(np.where(train_protected_info == 2)[0])

    val_data_standardized = loaded_test_data['standardized_data']
    val_protected_info = loaded_test_data['protected_info']
    val_class_label = loaded_test_data['class_label']
    val_protected_indices = list(np.where(val_protected_info == 1)[0])

    if (technique == 'euclidean'):
        weights_euclidean = loaded_optimization_info['weights_euclidean']
        disc_scores_with_reject, rejected_info_indices = give_Lenders_disc_score_with_reject_option(k, reject_threshold, train_class_label, train_unprotected_indices, \
                                                                                               train_data_standardized, val_protected_indices, val_class_label, val_data_standardized, indices_info, \
                                                                                               weighted_euclidean_distance, weights_euclidean)
    else:
        mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']
        disc_scores_with_reject, rejected_info_indices = give_Lenders_disc_score_with_reject_option(k, reject_threshold, train_class_label,
                                                                                                              train_unprotected_indices, train_data_standardized, val_protected_indices,
                                                                                                              val_class_label, val_data_standardized, indices_info, mahalanobis_distance,
                                                                                                              mahalanobis_matrix)
    return disc_scores_with_reject, rejected_info_indices
