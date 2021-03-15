from giving_discrimination_scores import give_all_disc_scores, give_disc_scores_with_reject_one_technique, give_disc_scores_one_technique
from load_data import load_data, load_optimization_info, split_test_sets
import pandas as pd
import numpy as np
import utils


def give_disc_scores_validation_set(location, indices_info, k_info, best_lambda_euclidean, best_lambda_mahalanobis):
    loaded_train_data = load_data(location, "train")
    loaded_val_data = load_data(location, "val")
    loaded_optimization_info = load_optimization_info(location, best_lambda_euclidean, best_lambda_mahalanobis)

    protected_info_val = loaded_val_data['protected_info']
    protected_indices_val = np.where(protected_info_val==1)[0]

    baseline_scores, luong_scores, zhang_scores, weighted_euclidean_scores, mahalanobis_scores = give_all_disc_scores(
        loaded_train_data, loaded_val_data,
        loaded_optimization_info, indices_info, k_info, "adult")

    data_frame_protected_indices_disc_scores = pd.DataFrame(list(zip(baseline_scores, luong_scores, zhang_scores, weighted_euclidean_scores, mahalanobis_scores)), index=protected_indices_val, columns=['Baseline', 'Luong', 'Zhang', 'Weighted Euclidean', 'Mahalanobis'])
    return data_frame_protected_indices_disc_scores


def make_diff_in_disc_scores_df(location, indices_info, k_info, best_lambda_euclidean, best_lambda_mahalanobis):
    dataframe_disc_scores = give_disc_scores_validation_set(location, indices_info, k_info, best_lambda_euclidean, best_lambda_mahalanobis)
    print(dataframe_disc_scores)
    column_names = dataframe_disc_scores.columns

    for index in range(len(column_names)):
        for second_index in range(index+1, len(column_names)):
            if (index != dataframe_disc_scores.shape[1]):
                name_diff_column = "diff_" + column_names[index] + "_" + column_names[second_index]
                dataframe_disc_scores[name_diff_column] = dataframe_disc_scores.iloc[:,index] - dataframe_disc_scores.iloc[:,second_index]

    utils.store_in_excel(dataframe_disc_scores, "C:/Users/daphn/PycharmProjects/kNN Situation Testing/adult/", "diff_in_disc_scores.xlsx")
    return dataframe_disc_scores


def correlations_disc_scores(location, indices_info, k_info, best_lambda_euclidean, best_lambda_mahalanobis):
    dataframe_disc_scores = give_disc_scores_validation_set(location, indices_info, k_info, best_lambda_euclidean, best_lambda_mahalanobis)
    correlations = dataframe_disc_scores.corr()
    print(correlations)
    return correlations


