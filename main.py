import numpy as np
import math
from visualize import visualize_luong, visualize_euclidean, visualize_mahalanobis, visualize_baseline
from test_discrimination_detection import area_under_curve_all_approaches, area_under_curve_test_sets, area_under_curve_validation_set
from perform_optimization import perform_optimization

if __name__ == '__main__':
    explainable_low_disc_location = "explainable-low-disc"
    explainable_medium_disc_location = "explainable-medium-disc"
    explainable_high_disc_location = "explainable-high-disc"

    non_explainable_medium_disc_location = "non-explainable-medium-disc"
    non_explainable_low_disc_location = "non-explainable-low-disc"
    non_explainable_high_disc_location = "non-explainable-high-disc"


    #perform_optimization(explainable_high_disc_location, 3100, 600, 10/31, 1/31)
    visualize_baseline(explainable_medium_disc_location, "Original Data")
    visualize_luong(explainable_medium_disc_location, "Original Standardized Data")
    visualize_euclidean(explainable_medium_disc_location, "Data Rescaled to Weighted Euclidean Distance")

    # print(area_under_curve_all_approaches(non_explainable_medium_disc_location, 20, 40))
    #area_under_curve_validation_set(explainable_high_disc_location, np.arange(10, 100, 10))
    #area_under_curve_test_sets(explainable_medium_disc_location, 10, 10, 20, 60)
    #
    # visualize_luong(load_non_explainable_high_discrimination_admission, "Original Standardized Data")
    # visualize_euclidean(load_non_explainable_high_discrimination_admission, "Data Rescaled to Weighted Euclidean Distance")
