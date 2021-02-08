import numpy as np
import pandas as pd
import math
from visualize import visualize_luong, visualize_euclidean, visualize_mahalanobis, visualize_baseline, visualize_positive_vs_negative
from test_discrimination_detection import area_under_curve_all_approaches, area_under_curve_test_sets, area_under_curve_validation_set,\
    find_best_threshold_and_k_based_on_val_F1, f1_score_test_sets
from perform_optimization import perform_optimization
from optimize_mahalanobis_distances import optimize_mahalanobis, mahalanobis_distance
from tuning_parameters import find_best_k_and_lambda_based_on_unprotected_region
import scipy

if __name__ == '__main__':
    data_non_explainable_low_disc_location = "non_explainable_low_disc"
    data_non_explainable_medium_disc_location = "non_explainable_medium_disc"

    data_explainable_low_disc_location = "explainable_low_disc"
    data_explainable_medium_disc_location = "explainable_medium_disc"
    #
    # possible_k_values = np.arange(10, 140, 10)
    # possible_threshold_values = np.arange(0.05, 0.4, 0.05)
    # # find_best_threshold_and_k_based_on_val_F1(big_data_explainable_low_disc_location, possible_k_values, possible_threshold_values)
    #
    k_info = {'baseline': 120, 'luong': 100, 'zhang': 100, 'euclidean': 10, 'mahalanobis': 10}
    # threshold_info = {'baseline': 0.35, 'luong': 0.35, 'zhang': 0.3, 'euclidean': 0.1}
    #
    # f1_score_test_sets(big_data_non_explainable_low_disc_location, 10, k_info, threshold_info)

    #juiste hoeveelheden 9000, 0.3, 10 / 18, 1 / 18
    # perform_optimization(data_non_explainable_medium_disc_location, 9000, 0.3, 10/18, 1/18)

    visualize_mahalanobis(data_non_explainable_medium_disc_location, 0.05, "Original Data")
    # visualize_luong(data_non_explainable_low_disc_location, "Original Standardized Data")
    # visualize_euclidean(data_explainable_low_disc_location, "Data Rescaled to Weighted Euclidean Distance")

    # find_best_k_and_threshold_based_on_man_region(data_non_explainable_low_disc_location, np.arange(10, 200, 10), 'luong')
    # find_best_k_and_lambda_based_on_unprotected_region(data_non_explainable_medium_disc_location, np.arange(10, 200, 10), np.arange(0.05, 0.35, 0.05), 'mahalanobis')
    # # print(area_under_curve_all_approaches(non_explainable_medium_disc_location, 20, 40))
    area_under_curve_validation_set(data_non_explainable_medium_disc_location, k_info=k_info, best_lambda_euclidean=0.2, best_lambda_mahalanobis=0.05)
    # area_under_curve_test_sets(data_non_explainable_medium_disc_location, 10, k_info=k_info, best_lambda_euclidean=0.2, best_lambda_mahalanobis=0.2)
    #
    # visualize_luong(load_non_explainable_high_discrimination_admission, "Original Standardized Data")
    # visualize_euclidean(load_non_explainable_high_discrimination_admission, "Data Rescaled to Weighted Euclidean Distance")
