from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


def train_classifier_on_unprotected_data(train_data, train_labels, train_unprotected_indices):

    X_unprotected_train = train_data.iloc[train_unprotected_indices]
    train_labels_unprotected = train_labels[train_unprotected_indices]

    random_forest = RandomForestClassifier(random_state=100)
    random_forest.fit(X_unprotected_train, train_labels_unprotected)

    return random_forest


def predict_test_data(test_data, trained_classifier):
    y_test_predicted = trained_classifier.predict_proba(test_data)[:,1]
    return y_test_predicted


def one_hot_encode(training_set, test_set):
    number_instances_in_training_set = len(training_set)
    dataset = pd.concat(objs=[training_set, test_set], axis=0)
    dataset_one_hot_encoded = pd.get_dummies(dataset)
    train_one_hot_encoded = dataset_one_hot_encoded[:number_instances_in_training_set]
    test_one_hot_encoded = dataset_one_hot_encoded[number_instances_in_training_set:]
    return train_one_hot_encoded, test_one_hot_encoded


def give_disc_scores_baseline(adult_or_admission, class_info_train, unprotected_indices_train, training_set, class_info_test, protected_indices_test, test_set):
    if (adult_or_admission=="adult"):
        training_set, test_set = one_hot_encode(training_set, test_set)

    trained_classifier = train_classifier_on_unprotected_data(training_set, class_info_train, unprotected_indices_train)

    protected_test_set = test_set.iloc[protected_indices_test]
    prediction_probability_protected_test_set = predict_test_data(protected_test_set, trained_classifier)

    positive_class_label_instances = np.where(class_info_test == 1)[0]
    positive_class_label_and_protected = list(set(protected_indices_test).intersection(set(positive_class_label_instances)))

    discrimination_scores = pd.DataFrame(data=prediction_probability_protected_test_set, index=protected_indices_test)
    discrimination_scores.loc[positive_class_label_and_protected] = -1

    return discrimination_scores.to_numpy().flatten().tolist()




