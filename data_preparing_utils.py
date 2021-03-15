from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import random
random.seed(6)


#order instances in dataset according to their prbability label
def order_instances(probability_labels):
    sort_by_probability = probability_labels.sort_values(1)
    sort_by_probability = sort_by_probability.reset_index(drop=True)
    return sort_by_probability


#learn classifier
def get_doubtful_cases(probability_labels, n_biased, class_labels):
    sorted_probability_labels = order_instances(probability_labels)
    discriminated_instances = []
    biased_counter = 0
    index_counter = 0
    while biased_counter<n_biased:
        index = sorted_probability_labels['Index'].iloc[index_counter]
        if class_labels[index] == 1:
            discriminated_instances.append(index)
            biased_counter+=1
        index_counter+=1
    return discriminated_instances



def learn_classifier(X_train, y_train):
    one_hot_encoded = pd.get_dummies(X_train)
    random_forest = RandomForestClassifier(random_state=100)
    random_forest.fit(one_hot_encoded, y_train)
    y_train_predict_probabilities = random_forest.predict_proba(one_hot_encoded)
    return y_train_predict_probabilities


def split_discriminated_indices(test_indices, train_indices, validation_data_indices, discriminated_indices):
    test_discriminated_indices = []
    train_discriminated_indices = []
    validation_discriminated_indices = []
    index_counter = 0
    for test_index in test_indices:
        if test_index in discriminated_indices:
            test_discriminated_indices.append(index_counter)
        index_counter +=1

    index_counter = 0
    for train_index in train_indices:
        if train_index in discriminated_indices:
            train_discriminated_indices.append(index_counter)
        index_counter +=1


    index_counter = 0
    for validation_index in validation_data_indices:
        if validation_index in discriminated_indices:
            validation_discriminated_indices.append(index_counter)
        index_counter +=1

    return train_discriminated_indices, test_discriminated_indices, validation_discriminated_indices
