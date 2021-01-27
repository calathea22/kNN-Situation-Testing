import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc

def store_in_excel(matrix, path):
    dataframe = pd.DataFrame(matrix)
    dataframe.to_excel(excel_writer=path)
    return

def give_separate_correlations(attribute, protected_class, dataset, variables_info, protected_label, unprotected_label):
    protected_indices = list(np.where(protected_class == protected_label)[0])
    unprotected_indices = list(np.where(protected_class == unprotected_label)[0])

    protected_data = dataset.iloc[protected_indices]
    unprotected_data = dataset.iloc[unprotected_indices]

    protected_label = attribute[protected_indices]
    unprotected_label = attribute[unprotected_indices]

    i = 0
    for columns in dataset.columns:
        if i in variables_info['interval'] or i in variables_info['ordinal']:
            print("Correlation to " + columns + " in general: " + str(stats.pearsonr(attribute, dataset[columns])[0]))
            print("Correlation to " + columns + " for protected: " + str(stats.pearsonr(protected_label, protected_data[columns])[0]))
            print("Correlation to " + columns + " for unprotected: " + str(stats.pearsonr(unprotected_label, unprotected_data[columns])[0]))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        i += 1
    return


def discriminated_instances_to_label_array(all_protected, instances):
    label_array = []
    for i in range(len(all_protected)):
        label_array.append(all_protected[i] in list(instances))
    return label_array


def interval_to_z_scores_train_set(dataset, indices_info):
    interval_vars = indices_info['interval']
    standardized_data = dataset.copy()

    for column in range(dataset.shape[1]):
        if column in interval_vars:
            standardized_data.iloc[:, column] = stats.zscore(standardized_data.iloc[:, column])
        else:
            standardized_data.iloc[:, column] = standardized_data.iloc[:, column]
    return standardized_data


def interval_to_z_scores_test_or_val_set(train_dataset, test_dataset, indices_info):
    interval_vars = indices_info['interval']
    standardized_data = test_dataset.copy()

    for column in range(test_dataset.shape[1]):
        if column in interval_vars:
            standardized_data.iloc[:, column] = (standardized_data.iloc[:, column] - train_dataset.iloc[:, column].mean()) / train_dataset.iloc[:, column].std(ddof=0)
        else:
            standardized_data.iloc[:, column] = standardized_data.iloc[:, column]
    return standardized_data


def get_auc_score(ground_truth, discrimination_score):
    fpr, tpr, thresholds = roc_curve(ground_truth, discrimination_score, pos_label=1)
    auc_score = auc(fpr, tpr)
    return auc_score


def bin_all_attributes_train(data, bin_dictionary):
    bin_edges_dict = {}
    new_data = data.copy()
    for column, number_of_bins in bin_dictionary.items():
        cut_labels = range(1, number_of_bins + 1)
        new_data['temp'], bin_edges = pd.cut(data[column], bins=number_of_bins, labels=cut_labels, retbins=True)
        bin_edges_dict[column] = bin_edges
        new_data[column] = new_data['temp']
        new_data = new_data.drop(columns=['temp'])
    return new_data, bin_edges_dict



def bin_based_on_train_bins(data, bin_edges_dictionary):
    new_data = data.copy()
    for column, bin_edges in bin_edges_dictionary.items():
        cut_labels = range(1, len(bin_edges))
        new_data['temp']= pd.cut(data[column], bins=bin_edges, labels=cut_labels)
        new_data[column] = new_data['temp']
        new_data = new_data.drop(columns=['temp'])
        new_data = new_data.replace(np.nan, 1)
    return new_data


def make_range_dict(data, indices_info):
    interval_columns = indices_info['interval']
    ordinal_columns = indices_info['ordinal']
    range_dict = {}
    if interval_columns!=None:
        for attribute in interval_columns:
            range_dict[attribute] = data[attribute].max()-data[attribute].min()
    if ordinal_columns!=None:
        for attribute in ordinal_columns:
            range_dict[attribute] = data[attribute].max()-data[attribute].min()
    return range_dict