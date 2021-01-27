import utils
import numpy as np
from optimize_distances import optimize_weighted_euclidean
from simulate_data import simulate_non_explainable_discrimination, simulate_explainable_discrimination


def perform_optimization(location, n, n_biased, test_percentage, val_percentage):
    train_data_dict, test_data_dict, val_data_dict = simulate_explainable_discrimination(n, n_biased, test_percentage, val_percentage)
    indices_info = {'interval': [0, 1, 2], 'ordinal': [], 'nominal': []}

    train_data = train_data_dict['data']
    train_protected_info = train_data_dict['protected_info']
    train_class_label = train_data_dict['class_label']
    train_discriminated_instances = train_data_dict['discriminated']

    standardized_train_data = np.array(utils.interval_to_z_scores_train_set(train_data, indices_info))

    val_data = val_data_dict['data']
    standardized_val_data = np.array(utils.interval_to_z_scores_test_or_val_set(train_data, val_data, indices_info))

    test_data = test_data_dict['data']
    standardized_test_data = np.array(utils.interval_to_z_scores_test_or_val_set(train_data, test_data, indices_info))

    weights_euclidean = optimize_weighted_euclidean(standardized_train_data, train_class_label, train_protected_info, indices_info, 1, 2)

    utils.store_in_excel(weights_euclidean,
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/euclidean_weights.xlsx")

    utils.store_in_excel(train_data,
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/" + location + "/train/data.xlsx")
    utils.store_in_excel(standardized_train_data,
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/train/standardized_data.xlsx")
    utils.store_in_excel(train_protected_info,
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/train/protected_attribute.xlsx")
    utils.store_in_excel(train_class_label,
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/train/class_label.xlsx")
    utils.store_in_excel(train_discriminated_instances,
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/train/discriminated_instances.xlsx")

    utils.store_in_excel(test_data,
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/" + location + "/test/data.xlsx")
    utils.store_in_excel(standardized_test_data,
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/test/standardized_data.xlsx")
    utils.store_in_excel(test_data_dict['protected_info'],
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/test/protected_attribute.xlsx")
    utils.store_in_excel(test_data_dict['class_label'],
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/test/class_label.xlsx")
    utils.store_in_excel(test_data_dict['discriminated'],
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/test/discriminated_instances.xlsx")

    utils.store_in_excel(val_data,
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/" + location + "/val/data.xlsx")
    utils.store_in_excel(standardized_val_data,
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/val/standardized_data.xlsx")
    utils.store_in_excel(val_data_dict['protected_info'],
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/val/protected_attribute.xlsx")
    utils.store_in_excel(val_data_dict['class_label'],
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/val/class_label.xlsx")
    utils.store_in_excel(val_data_dict['discriminated'],
                         "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"+location+"/val/discriminated_instances.xlsx")
    return


