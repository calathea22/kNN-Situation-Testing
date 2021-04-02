import utils
import numpy as np
import pandas as pd
from optimize_euclidean_distance import optimize_weighted_euclidean
from optimize_mahalanobis_distance import optimize_mahalanobis
from data_preparing_simulation import simulate_explainable_discrimination_admission_data, simulate_non_explainable_discrimination_admission_data, simulate_train_test_and_val_with_discrimination, \
    simulate_non_explainable_discrimination_4_attributes_admission_data
from data_preparing_adult import load_adult_train_val_test_data

def perform_optimization(location, simulation_function, n, test_percentage, val_percentage, n_biased=0):
    parent_path = "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"

    train_data_dict, test_data_dict, val_data_dict = load_adult_train_val_test_data(n, test_percentage, val_percentage)
    #train_data_dict, test_data_dict, val_data_dict = simulate_train_test_and_val_with_discrimination(n, n_biased, test_percentage, val_percentage, simulation_function)

    #indices_info = {'interval': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'ordinal': []}
    indices_info = train_data_dict['indices_info']
    print(indices_info)

    train_data = train_data_dict['data']
    train_protected_info = train_data_dict['protected_info']
    train_class_label = train_data_dict['class_label']


    standardized_train_data = np.array(utils.interval_to_z_scores_train_set(train_data, indices_info))

    val_data = val_data_dict['data']
    standardized_val_data = np.array(utils.interval_to_z_scores_test_or_val_set(train_data, val_data, indices_info))

    test_data = test_data_dict['data']
    standardized_test_data = np.array(utils.interval_to_z_scores_test_or_val_set(train_data, test_data, indices_info))

    train_path = parent_path + location + "/train"
    val_path = parent_path + location + "/val"
    test_path = parent_path + location + "/test"

    utils.store_in_excel(train_data, train_path, "/data.xlsx")
    utils.store_in_excel(standardized_train_data, train_path, "/standardized_data.xlsx")
    utils.store_in_excel(train_protected_info, train_path, "/protected_attribute.xlsx")
    utils.store_in_excel(train_class_label, train_path, "/class_label.xlsx")

    utils.store_in_excel(test_data, test_path, "/data.xlsx")
    utils.store_in_excel(standardized_test_data, test_path, "/standardized_data.xlsx")
    utils.store_in_excel(test_data_dict['protected_info'], test_path, "/protected_attribute.xlsx")
    utils.store_in_excel(test_data_dict['class_label'], test_path, "/class_label.xlsx")

    utils.store_in_excel(val_data, val_path, "/data.xlsx")
    utils.store_in_excel(standardized_val_data, val_path, "/standardized_data.xlsx")
    utils.store_in_excel(val_data_dict['protected_info'], val_path, "/protected_attribute.xlsx")
    utils.store_in_excel(val_data_dict['class_label'], val_path, "/class_label.xlsx")

    if (n_biased != 0):
        utils.store_in_excel(train_data_dict['discriminated'], train_path, "/discriminated_instances.xlsx")
        utils.store_in_excel(train_data_dict['disc_free_labels'], train_path, "/disc_free_labels.xlsx")

        utils.store_in_excel(test_data_dict['discriminated'], test_path, "/discriminated_instances.xlsx")
        utils.store_in_excel(test_data_dict['disc_free_labels'], test_path, "/disc_free_labels.xlsx")

        utils.store_in_excel(val_data_dict['discriminated'], val_path, "/discriminated_instances.xlsx")
        utils.store_in_excel(val_data_dict['disc_free_labels'], val_path, "/disc_free_labels.xlsx")

    #possible_lambdas = np.arange(0.09, 0.02, -0.02)

    possible_lambdas = [0.07, 0.05, 0.09]
    for lambda_l1_norm in possible_lambdas:
        weights_euclidean = optimize_weighted_euclidean(standardized_train_data, train_class_label, train_protected_info, indices_info, 1, 2, lambda_l1_norm)
        print(weights_euclidean)

        utils.store_in_excel(weights_euclidean,
                             parent_path + location+"/lambda = " +format(lambda_l1_norm, ".3f"), "/euclidean_weights.xlsx")

        mahalanobis_matrix = optimize_mahalanobis(standardized_train_data, train_class_label, train_protected_info,
                                                  indices_info, 1, 2, lambda_l1_norm)
        print(mahalanobis_matrix)
        utils.store_in_excel(mahalanobis_matrix,
                             parent_path + location + "/lambda = " +format(lambda_l1_norm, ".3f"), "/mahalanobis_matrix.xlsx")

    return




def perform_optimization_from_loaded_data(location):
    parent_path = "C:/Users/daphn/PycharmProjects/kNN Situation Testing/"

    indices_info = {'interval': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'ordinal': []}

    data = pd.read_excel(
        parent_path + location + "/" + "train" + "/data.xlsx")
    standardized_data = pd.read_excel(
        parent_path + location + "/" + "train" + "/standardized_data.xlsx")
    class_label = pd.read_excel(
        parent_path + location + "/" + "train" + "/class_label.xlsx")
    protected_info = pd.read_excel(
        parent_path + location + "/" + "train" + "/protected_attribute.xlsx")

    data = data.drop('Unnamed: 0', axis='columns')
    train_class_label = np.array(class_label.drop('Unnamed: 0', axis='columns')[0])
    standardized_train_data = np.array(standardized_data.drop('Unnamed: 0', axis='columns'))
    train_protected_info = np.array(protected_info.drop('Unnamed: 0', axis='columns')[0])

    possible_lambdas = np.arange(0.4, 0.04, -0.02)
    for lambda_l1_norm in possible_lambdas:
        weights_euclidean = optimize_weighted_euclidean(standardized_train_data, train_class_label, train_protected_info, indices_info, 1, 2, lambda_l1_norm)
        print(weights_euclidean)

        utils.store_in_excel(weights_euclidean,
                             parent_path + location+"/lambda = " +format(lambda_l1_norm, ".3f"), "/euclidean_weights.xlsx")

        mahalanobis_matrix = optimize_mahalanobis(standardized_train_data, train_class_label, train_protected_info,
                                                  indices_info, 1, 2, lambda_l1_norm)
        print(mahalanobis_matrix)
        utils.store_in_excel(mahalanobis_matrix,
                             parent_path + location + "/lambda = " +format(lambda_l1_norm, ".3f"), "/mahalanobis_matrix.xlsx")

    return
