import pandas as pd
import numpy as np
import utils
from load_data import load_data, load_optimization_info
from kNN_discrimination_discovery import give_all_disc_scores_euclidean, give_all_disc_scores_Luong
from sklearn.metrics import confusion_matrix


def relate_label_correctness_to_score_info(score_info, disc_labels, ground_truth, data):
    correct_labels_score_info = []
    incorrect_labels_score_info = []
    for i in range(len(disc_labels)):
        if disc_labels[i] == ground_truth[i]:
            correct_labels_score_info.append(score_info[i])
        else:
            incorrect_labels_score_info.append(score_info[i])
            print(data.iloc[score_info[i][0]])

    return


def inspect_mistakes(location, k, threshold):
    loaded_train_data = load_data(location, "train")
    loaded_val_data = load_data(location, "val")
    loaded_optimization_info = load_optimization_info(location)

    train_data = loaded_train_data['data']
    train_data_standardized = loaded_train_data['standardized_data']
    train_protected_info = loaded_train_data['protected_info']
    train_class_label = loaded_train_data['class_label']
    train_protected_indices = list(np.where(train_protected_info == 1)[0])
    train_unprotected_indices = list(np.where(train_protected_info == 2)[0])

    val_data = loaded_val_data['data']
    val_data_standardized = loaded_val_data['standardized_data']
    val_ground_truth = loaded_val_data['ground_truth']
    val_protected_info = loaded_val_data['protected_info']
    val_class_label = loaded_val_data['class_label']
    val_protected_indices = list(np.where(val_protected_info == 1)[0])

    indices_info = loaded_optimization_info['indices_info']
    weights_euclidean = loaded_optimization_info['weights_euclidean']

    discrimination_scores =  give_all_disc_scores_Luong(k, class_info_train=train_class_label,
                                              protected_indices_train=train_protected_indices,
                                              unprotected_indices_train=train_unprotected_indices,
                                              training_set=train_data_standardized,
                                              protected_indices_test=val_protected_indices,
                                              class_info_test=val_class_label, test_set=val_data_standardized,
                                              indices_info=indices_info)
    disc_labels = utils.give_disc_label(discrimination_scores, threshold)

    print(confusion_matrix(val_ground_truth, disc_labels))
    # print("Precision: " )

    # relate_label_correctness_to_score_info(disc_score_info, disc_labels, val_ground_truth, val_data)



