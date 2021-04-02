import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import math
import os
from matplotlib import pyplot
from scipy.spatial.distance import pdist, squareform, cdist


def store_in_excel(matrix, path, filename):
    dataframe = pd.DataFrame(matrix)

    if not os.path.exists(path):
        os.makedirs(path)

    dataframe.to_excel(excel_writer=path+filename)


    # dataframe.to_excel(excel_writer=path)
    return


def give_description_of_nearest_neighbours(neighbour_data, data_of_instance_in_question, indices_info):
    for interval_index in indices_info['interval']:
        column_name = neighbour_data.columns[interval_index]
        column_from_data = neighbour_data.iloc[:, interval_index]
        print(column_name)
        print("Value of instance on this feature: " + str(data_of_instance_in_question.iloc[interval_index]))
        print("Mean value of neighbours for this feature: " + str(column_from_data.mean()))
        print("Standard deviation of neighbours for this feature: " + str(column_from_data.std()))

    # column_name = neighbour_data.columns[1]
    # column_from_data = neighbour_data.iloc[:, 1]
    # print(column_name)
    # print("Value of instance on this feature: " + str(data_of_instance_in_question.iloc[1]))
    # print("Mean value of neighbours for this feature: " + str(column_from_data.mean()))

    for ordinal_index in indices_info['ordinal']:

        column_name = neighbour_data.columns[ordinal_index]
        print(column_name)
        column_from_data = neighbour_data.iloc[:, ordinal_index]
        print("Value of instance on this feature: " + str(data_of_instance_in_question.iloc[ordinal_index]))
        print(neighbour_data.groupby(column_name).count())



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


def discriminated_instances_to_label_array(all_protected, discriminated_instances):
    label_array = []
    for i in range(len(all_protected)):
        label_array.append(all_protected[i] in list(discriminated_instances))
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


def get_auc_scores(ground_truth, discrimination_score):
    precision, recall, thresholds_p_r = precision_recall_curve(ground_truth, discrimination_score)
    auc_score_pr = auc(recall, precision)
    fpr, tpr, thresholds_roc = roc_curve(ground_truth, discrimination_score, pos_label=1)
    auc_score_roc = auc(fpr, tpr)
    return auc_score_pr, auc_score_roc


def get_roc_auc_score(ground_truth, discrimination_score):
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


def give_disc_label(discrimination_scores, threshold):
    discrimination_labels = []
    for discrimination_score in discrimination_scores:
        discrimination_labels.append(discrimination_score >= threshold)
    return discrimination_labels


def print_avg_results_from_dictionary(results_dictionary):
    baseline_results = [result['baseline'] for result in results_dictionary]
    print("Baseline: " + str(sum(baseline_results) / len(baseline_results)))
    luong_results = [result['luong'] for result in results_dictionary]
    print("Luong: " + str(sum(luong_results) / len(luong_results)))
    zhang_results = [result['zhang'] for result in results_dictionary]
    print("Zhang: " + str(sum(zhang_results) / len(zhang_results)))
    euclidean_results = [result['euclidean'] for result in results_dictionary]
    print("Euclidean: " + str(sum(euclidean_results) / len(euclidean_results)))
    mahalanobis_results = [result['mahalanobis'] for result in results_dictionary]
    print("Mahalanobis: " + str(sum(mahalanobis_results) / len(mahalanobis_results)))
    return

def project_to_weighted_euclidean(data, weights):
    for i in range(len(weights)):
        data.iloc[:,i] = data.iloc[:,i] * math.sqrt(weights[i])
    return data


def project_to_mahalanobis(data, mahalanobis_matrix):
    eigvals, eigvecs = np.linalg.eigh(mahalanobis_matrix)
    print(eigvals)
    eigvals = eigvals.astype(float)  # Remove residual imaginary part
    eigvecs = eigvecs.astype(float)
    eigvals[eigvals < 0.0] = 0.0
    sqrt_diag = np.sqrt(eigvals)
    L = eigvecs.dot(np.diag(sqrt_diag)).T
    projected_data = pd.DataFrame(data.dot(L.T))
    return projected_data


# given the indices with the discriminated labels in the dataset, this function turns the corresponding class labels form
# negative to positive
def generate_non_discriminated_class_info(ground_truth_labels, protected_info, class_labels, protected_label):
    new_class_labels = []
    discriminated_index_counter = 0
    for i in range(len(class_labels)):
        if protected_info[i] == protected_label:
            if ground_truth_labels[discriminated_index_counter] == 1:
                new_class_labels.append(True)
            else:
                new_class_labels.append(class_labels.iloc[i])
            discriminated_index_counter+=1
        else:
            new_class_labels.append(class_labels.iloc[i])
    return new_class_labels


# def remove_discrimination_from_protected_indices(discrimination_labels, biased_class_labels):
#     new_class_labels = []
#     for i in range(len(biased_class_labels)):
#         if i<len(discrimination_labels):
#             if discrimination_labels[i] == True:
#                 new_class_labels.append(True)
#         else:
#             new_class_labels.append(biased_class_labels.iloc[i])
#     return new_class_labels

def remove_discrimination_from_protected_indices(discrimination_labels, biased_class_labels, protected_indices):
    new_class_labels = []
    protected_index_counter = 0
    for i in range(len(biased_class_labels)):
        if i not in protected_indices:
            new_class_labels.append(biased_class_labels.iloc[i])
        else:
            new_class_labels.append(discrimination_labels[protected_index_counter])
            protected_index_counter += 1
    return new_class_labels




def get_inter_and_intra_sens_distances(distances, sens_attribute, prot_label):
    inter_prot = []
    inter_unprot = []
    intra = []
    for i in range(0, len(sens_attribute)):
        for j in range(i + 1, len(sens_attribute)):
            if sens_attribute[i] != sens_attribute[j]:
                intra.append(distances.iloc[i, j])
            elif sens_attribute[i] == prot_label:
                inter_prot.append(distances.iloc[i, j])
            else:
                inter_unprot.append(distances.iloc[i, j])

    end_of_prot_indices = len(inter_prot)

    beginning_of_unprot_indices = len(inter_prot)
    end_of_unprot_indices = beginning_of_unprot_indices + len(inter_unprot)

    beginning_of_intra_indices = end_of_unprot_indices

    all_distances = inter_prot + inter_unprot + intra
    all_distances_normalized = (all_distances - min(all_distances)) / (max(all_distances) - min(all_distances))

    normalized_inter_prot = all_distances_normalized[0:end_of_prot_indices]
    normalized_inter_unprot = all_distances_normalized[beginning_of_unprot_indices:end_of_unprot_indices]
    normalized_intra = all_distances_normalized[beginning_of_intra_indices:len(all_distances)]

    # boxplot(data=[normalized_inter_prot, normalized_inter_unprot, normalized_intra])
    # plt.show()
    print("Mean inter protected:")
    print(sum(normalized_inter_prot)/len(normalized_inter_prot))
    print("Mean inter unprotected:")
    print(sum(normalized_inter_unprot)/len(normalized_inter_unprot))
    print("Mean intra")
    print(sum(normalized_intra)/len(normalized_intra))
    return normalized_inter_prot, normalized_inter_unprot, normalized_intra


def make_distance_matrix_based_on_distance_function(data, distance_function, weights, indices_info):
    dists = pdist(data, distance_function, weights=weights, indices_info=indices_info)
    distance_matrix = pd.DataFrame(squareform(dists))
    return distance_matrix