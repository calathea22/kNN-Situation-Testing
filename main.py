import numpy as np
import pandas as pd
import math
from test_discrimination_detection import area_under_curve_all_approaches, area_under_curve_test_sets, area_under_curve_validation_set,\
     f1_score_test_sets, area_under_curve_validation_set_different_k, f1_score_val_set_different_thresholds
from test_discrimination_detection_without_ground_truth import make_diff_in_disc_scores_df, comparing_algorithms_on_specific_indices
from perform_optimization import perform_optimization
from tuning_parameters import find_best_k_based_on_unprotected_region, find_best_k_and_lambda_based_on_unprotected_region, find_best_threshold_based_on_unprotected_region
import utils
from data_preparing_simulation import simulate_explainable_discrimination_admission_data, simulate_non_explainable_discrimination_admission_data, simulate_train_test_and_val_with_discrimination, \
    simulate_non_explainable_discrimination_4_attributes_admission_data, simulate_non_explainable_discrimination_7_attributes_admission_data, simulate_explainable_discrimination_4_attributes_admission_data, \
    simulate_explainable_discrimination_7_attributes_admission_data

if __name__ == '__main__':
    non_explainable_medium_disc_location = "non_explainable_medium_disc"
    non_explainable_medium_disc_4_irrelevant_attributes = "non_explainable_medium_disc_4_irrelevant_attributes"
    non_explainable_medium_disc_7_irrelevant_attributes = "non_explainable_medium_disc_7_irrelevant_attributes"

    explainable_low_disc_location = "explainable_low_disc"
    explainable_low_disc_4_irrelevant_attributes = "explainable_low_disc_4_irrelevant_attributes"
    explainable_low_disc_7_irrelevant_attributes = "explainable_low_disc_7_irrelevant_attributes"

    adult_without_groundtruth = "adult_without_groundtruth"

    indices_info_admission = {'interval': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'ordinal': []}
    indices_info_adult_final = {'interval': [0, 3, 4, 2], 'nominal': [], 'ordinal': [1, 5, 6, 7, 8, 9, 10, 11, 12]}

    k_info_non_explainable = {'luong': 10, 'zhang': 30, 'euclidean': 10, 'mahalanobis': 20}
    k_info_explainable = {'luong': 10, 'zhang': 10, 'euclidean': 10, 'mahalanobis': 20}
    k_info_adult = {'luong': 10, 'zhang': 10, 'euclidean': 10, 'mahalanobis': 10}


    area_under_curve_test_sets(non_explainable_medium_disc_location, indices_info_admission, 10, k_info=k_info_non_explainable, best_lambda_euclidean=0.07, best_lambda_mahalanobis=0.09, adult_or_admission="admission")
