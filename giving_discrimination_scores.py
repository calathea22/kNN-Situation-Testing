import numpy as np
from baseline_disc_detection import give_disc_scores_baseline
from kNN_discrimination_discovery import give_all_disc_scores_Luong, give_disc_score_k_approach, \
    weighted_euclidean_distance, mahalanobis_distance, luong_distance, give_all_disc_scores_euclidean_2k_approach, \
    give_all_disc_scores_mahalanobis_2k_approach, give_k_disc_score_with_info_about_nearest_neighbours
from perform_zhang_algorithm import get_zhang_discrimination_scores
import utils
from load_data import load_data, load_optimization_info


def give_disc_scores_to_given_indices(loaded_train_data, loaded_test_data, loaded_optimization_info, indices_info, k_info, adult_or_admission, indices_of_interest):
    train_data = loaded_train_data['data']
    train_data_standardized = loaded_train_data['standardized_data']
    train_protected_info = loaded_train_data['protected_info']
    train_class_label = loaded_train_data['class_label']
    train_unprotected_indices = list(np.where(train_protected_info == 2)[0])

    val_data = loaded_test_data['data']
    val_data_standardized = loaded_test_data['standardized_data']
    val_protected_info = loaded_test_data['protected_info']
    val_class_label = loaded_test_data['class_label']

    weights_euclidean = loaded_optimization_info['weights_euclidean']
    mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']

    luong_scores_1k, _ = give_k_disc_score_with_info_about_nearest_neighbours(k=k_info['luong'], class_info_train=train_class_label,
                                                                              unprotected_indices_train=train_unprotected_indices, standardized_training_set=train_data_standardized, training_set= train_data, protected_indices_test=indices_of_interest,
                                                                              class_info_test=val_class_label, standardized_test_set=val_data_standardized, test_set=val_data, indices_info=indices_info, distance_function=luong_distance,
                                                                              weights=[])

    zhang_scores = get_zhang_discrimination_scores(adult_or_admission, k=k_info['zhang'], train_data=train_data,
                                                             train_sens_attribute=train_protected_info,
                                                             train_decision_attribute=train_class_label,
                                                             test_data=val_data,
                                                             test_sens_attribute=val_protected_info,
                                                             test_decision_attribute=val_class_label, protected_indices=indices_of_interest)

    weighted_euclidean_scores, _ = give_k_disc_score_with_info_about_nearest_neighbours(k=k_info['euclidean'], class_info_train=train_class_label,
                                                                                        unprotected_indices_train=train_unprotected_indices, standardized_training_set=train_data_standardized, training_set= train_data, protected_indices_test=indices_of_interest,
                                                                                        class_info_test=val_class_label, standardized_test_set=val_data_standardized, test_set=val_data, indices_info=indices_info, distance_function=weighted_euclidean_distance,
                                                                                        weights=weights_euclidean)


    mahalanobis_scores, _ = give_k_disc_score_with_info_about_nearest_neighbours(k=k_info['mahalanobis'], class_info_train=train_class_label,
                                                                                 unprotected_indices_train=train_unprotected_indices, standardized_training_set=train_data_standardized, training_set= train_data, protected_indices_test=indices_of_interest,
                                                                                 class_info_test=val_class_label, standardized_test_set=val_data_standardized, test_set=val_data, indices_info=indices_info, distance_function=mahalanobis_distance,
                                                                                 weights=mahalanobis_matrix)
    return luong_scores_1k, zhang_scores, weighted_euclidean_scores, mahalanobis_scores


def give_all_disc_scores_k_approach(loaded_train_data, loaded_test_data, loaded_optimization_info, indices_info, k_info, adult_or_admission):
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

    weights_euclidean = loaded_optimization_info['weights_euclidean']
    mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']

    #
    baseline_scores = give_disc_scores_baseline(adult_or_admission, class_info_train=train_class_label, unprotected_indices_train=train_unprotected_indices,
                                                training_set=train_data, class_info_test=val_class_label,
                                                protected_indices_test=val_protected_indices, test_set=val_data)

    luong_scores_1k, _ = give_disc_score_k_approach(k=k_info['luong'], class_info_train=train_class_label,
                                                    unprotected_indices_train=train_unprotected_indices, training_set=train_data_standardized, protected_indices_test=val_protected_indices,
                                                    class_info_test=val_class_label, test_set=val_data_standardized, indices_info=indices_info, distance_function=luong_distance,
                                                    weights=[])

    #to change k-approach here go into code of zhang algorithm
    zhang_scores = get_zhang_discrimination_scores(adult_or_admission, k=k_info['zhang'], train_data=train_data,
                                                             train_sens_attribute=train_protected_info,
                                                             train_decision_attribute=train_class_label,
                                                             test_data=val_data,
                                                             test_sens_attribute=val_protected_info,
                                                             test_decision_attribute=val_class_label, protected_indices=val_protected_indices)

    weighted_euclidean_scores_1k, _ = give_disc_score_k_approach(k=k_info['euclidean'], class_info_train=train_class_label,
                                                                 unprotected_indices_train=train_unprotected_indices, training_set=train_data_standardized, protected_indices_test=val_protected_indices,
                                                                 class_info_test=val_class_label, test_set=val_data_standardized, indices_info=indices_info, distance_function=weighted_euclidean_distance,
                                                                 weights=weights_euclidean)


    mahalanobis_scores_1k, _ = give_disc_score_k_approach(k=k_info['mahalanobis'], class_info_train=train_class_label,
                                                          unprotected_indices_train=train_unprotected_indices, training_set=train_data_standardized, protected_indices_test=val_protected_indices,
                                                          class_info_test=val_class_label, test_set=val_data_standardized, indices_info=indices_info, distance_function=mahalanobis_distance,
                                                          weights=mahalanobis_matrix)


    return baseline_scores, luong_scores_1k, zhang_scores, weighted_euclidean_scores_1k, mahalanobis_scores_1k



def give_all_disc_scores_k_plus_k_approach(loaded_train_data, loaded_test_data, loaded_optimization_info, indices_info, k_info, adult_or_admission):
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

    weights_euclidean = loaded_optimization_info['weights_euclidean']
    mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']

    #baseline doesn't change, (since it's not based on situation testing we do not differentiate between k and k plus k approach
    baseline_scores = give_disc_scores_baseline(adult_or_admission, class_info_train=train_class_label, unprotected_indices_train=train_unprotected_indices,
                                                training_set=train_data, class_info_test=val_class_label,
                                                protected_indices_test=val_protected_indices, test_set=val_data)

    luong_scores_2k = give_all_disc_scores_Luong(k_info['luong'], class_info_train=train_class_label,
                                              protected_indices_train=train_protected_indices,
                                              unprotected_indices_train=train_unprotected_indices,
                                              training_set=train_data_standardized,
                                              protected_indices_test=val_protected_indices,
                                              class_info_test=val_class_label, test_set=val_data_standardized,
                                              indices_info=indices_info)

    #to change k-approach here go into code of zhang algorithm
    zhang_scores = get_zhang_discrimination_scores(adult_or_admission, k=k_info['zhang'], train_data=train_data,
                                                             train_sens_attribute=train_protected_info,
                                                             train_decision_attribute=train_class_label,
                                                             test_data=val_data,
                                                             test_sens_attribute=val_protected_info,
                                                             test_decision_attribute=val_class_label, protected_indices=val_protected_indices)

    weighted_euclidean_scores_2k = give_all_disc_scores_euclidean_2k_approach(k=k_info['euclidean'], class_info_train=train_class_label, protected_indices_train= train_protected_indices,
                         unprotected_indices_train=train_unprotected_indices, training_set=train_data_standardized, protected_indices_test=val_protected_indices,
                         class_info_test=val_class_label, test_set=val_data_standardized, indices_info=indices_info, weights=weights_euclidean)


    mahalanobis_scores_2k = give_all_disc_scores_mahalanobis_2k_approach(k=k_info['mahalanobis'], class_info_train=train_class_label, protected_indices_train= train_protected_indices,
                         unprotected_indices_train=train_unprotected_indices, training_set=train_data_standardized, protected_indices_test=val_protected_indices,
                         class_info_test=val_class_label, test_set=val_data_standardized, indices_info=indices_info, weight_matrix=mahalanobis_matrix)

    return baseline_scores, luong_scores_2k, zhang_scores, weighted_euclidean_scores_2k, mahalanobis_scores_2k


def give_disc_scores_one_technique(loaded_train_data, loaded_test_data, loaded_optimization_info, technique, indices_info, k,  protected_info_label_of_interest=1, adult_or_admission="admission"):
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
    val_protected_indices = list(np.where(val_protected_info == protected_info_label_of_interest)[0])

    if technique == 'baseline':
        baseline_scores = give_disc_scores_baseline(adult_or_admission, class_info_train=train_class_label,
                                                    unprotected_indices_train=train_unprotected_indices,
                                                    training_set=train_data, class_info_test=val_class_label,
                                                    protected_indices_test=val_protected_indices, test_set=val_data)
        return baseline_scores
    elif technique == 'luong':
        luong_scores, _ = give_disc_score_k_approach(k, class_info_train=train_class_label,
                                                     unprotected_indices_train=train_unprotected_indices,
                                                     training_set=train_data_standardized,
                                                     protected_indices_test=val_protected_indices,
                                                     class_info_test=val_class_label,
                                                     test_set=val_data_standardized,
                                                     indices_info=indices_info,
                                                     distance_function=luong_distance,
                                                     weights=[])
        return luong_scores
    elif technique == 'zhang':
        zhang_scores = get_zhang_discrimination_scores(adult_or_admission, k=k, train_data=train_data,
                                                       train_sens_attribute=train_protected_info,
                                                       train_decision_attribute=train_class_label,
                                                       test_data=val_data,
                                                       test_sens_attribute=val_protected_info,
                                                       test_decision_attribute=val_class_label,
                                                       protected_indices=val_protected_indices)
        return zhang_scores
    elif technique == 'euclidean':
        weights_euclidean = loaded_optimization_info['weights_euclidean']
        weighted_euclidean_scores, _ = give_disc_score_k_approach(k, class_info_train=train_class_label,
                                                                  unprotected_indices_train=train_unprotected_indices,
                                                                  training_set=train_data_standardized,
                                                                  protected_indices_test=val_protected_indices,
                                                                  class_info_test=val_class_label,
                                                                  test_set=val_data_standardized,
                                                                  indices_info=indices_info,
                                                                  distance_function=weighted_euclidean_distance,
                                                                  weights=weights_euclidean)
        return weighted_euclidean_scores
    elif technique == 'mahalanobis':
        mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']
        mahalanobis_scores, _ = give_disc_score_k_approach(k, class_info_train=train_class_label,
                                                           unprotected_indices_train=train_unprotected_indices,
                                                           training_set=train_data_standardized,
                                                           protected_indices_test=val_protected_indices,
                                                           class_info_test=val_class_label,
                                                           test_set=val_data_standardized,
                                                           indices_info=indices_info,
                                                           distance_function=mahalanobis_distance,
                                                           weights=mahalanobis_matrix)
        return mahalanobis_scores

    return 0
