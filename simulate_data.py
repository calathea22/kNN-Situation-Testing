import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from scipy import stats


#order instances in dataset according to their prbability label
def order_instances(probability_labels):
    #probability_labels_credible_women = probability_labels[probability_labels[1] > 0.5]
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


def add_discrimination_admission(data_dict, n_biased):
    training_data = data_dict['training_data']
    gender = data_dict['protected_info']
    training_data['gender'] = gender
    class_labels = data_dict['class_label']

    predicted_probabilites = learn_classifier(training_data, class_labels)

    protected_indices = list(np.where(gender == 1)[0])
    predicted_probabilites_women = pd.DataFrame(predicted_probabilites[protected_indices])
    predicted_probabilites_women['Index'] = protected_indices

    discriminated_indices = get_doubtful_cases(predicted_probabilites_women, n_biased, class_labels)
    np.put(class_labels, discriminated_indices, False)

    training_data = training_data.drop(columns=['gender'])

    return ({'class_label': class_labels, 'protected_info': gender, 'training_data': training_data}, discriminated_indices)


def admission_data_get_columns_info():
    interval_vars = ['Height', 'Score', 'Extra Curricular']
    nominal_vars = []
    ordinal_vars = []

    columns_info = {'interval': interval_vars, 'nominal': nominal_vars, 'ordinal': ordinal_vars}
    return columns_info


def simulate_non_explainable_discrimination_admission_data(n):
    np.random.seed(6)
    gender = np.random.randint(2, size=n) + 1

    height = np.round(10 * gender + np.random.normal(170, 2, n))

    testscore = np.round(np.random.normal(5, 2, n))

    extra_curricular = np.round(np.random.normal(5, 2, n))

    admission = 3 * testscore + 2 * extra_curricular + np.random.normal(5, 2, n)
    admission_probabilities = (admission - min(admission)) / (max(admission) - min(admission))
    admission_labels = [admission_probability > 0.5 for admission_probability in admission_probabilities]

    dataframe = pd.DataFrame(list(zip(height, testscore, extra_curricular)),
                             columns=['Height', 'Score', 'Extra Curricular'])

    return {'class_label': np.array(admission_labels), 'protected_info': np.array(gender), 'training_data': dataframe}


def simulate_explainable_discrimination_admission_data(n):
    np.random.seed(6)
    gender = np.random.randint(2, size=n) + 1

    height = np.round(10 * gender + np.random.normal(170, 2, n))
    testscore = np.round(np.random.normal(5, 2, n))
    extra_curricular = 3*gender + np.round(np.random.normal(5, 2, n))
    print(extra_curricular)

    admission = 3 * testscore + 4 * extra_curricular + np.random.normal(5, 2, n)
    admission_probabilities = (admission - min(admission)) / (max(admission) - min(admission))
    admission_labels = [admission_probability > 0.5 for admission_probability in admission_probabilities]

    dataframe = pd.DataFrame(list(zip(height, testscore, extra_curricular)),
                             columns=['Height', 'Score', 'Extra Curricular'])
    return {'class_label': np.array(admission_labels), 'protected_info': np.array(gender), 'training_data': dataframe}



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


def split_data_dict(data_dict, discriminated_indices, n_samples_test, n_samples_val):
    complete_data = data_dict['training_data']
    complete_protected_info = data_dict['protected_info']
    complete_labels = data_dict['class_label']

    test_data = complete_data.sample(n=n_samples_test)
    test_data_indices = test_data.index
    protected_info_test = complete_protected_info[test_data_indices]
    labels_test = complete_labels[test_data_indices]

    train_data = complete_data.drop(test_data_indices)
    validation_data = train_data.sample(n=n_samples_val)
    validation_data_indices = validation_data.index
    protected_info_validation = complete_protected_info[validation_data_indices]
    labels_validation = complete_labels[validation_data_indices]

    train_data = train_data.drop(validation_data_indices)
    train_data_indices = train_data.index
    test_and_validation_indices = np.append(validation_data_indices, test_data_indices)
    protected_info_train = np.delete(complete_protected_info, test_and_validation_indices)
    labels_train = np.delete(complete_labels, test_and_validation_indices)
    train_discriminated_indices, test_discriminated_indices, validation_discriminated_indices = split_discriminated_indices(
        test_data_indices, train_data_indices, validation_data_indices, discriminated_indices)

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    validation_data.reset_index(drop=True, inplace=True)

    train_data_dict = {'data': train_data, 'protected_info': np.array(protected_info_train), 'class_label': np.array(labels_train),
                       'discriminated': train_discriminated_indices}
    test_data_dict = {'data': test_data, 'protected_info': protected_info_test, 'class_label': labels_test,
                      'discriminated': test_discriminated_indices}
    validation_data_dict = {'data': validation_data, 'protected_info': protected_info_validation,
                            'class_label': labels_validation,
                            'discriminated': validation_discriminated_indices}
    return train_data_dict, test_data_dict, validation_data_dict


def simulate_non_explainable_discrimination(n, n_biased, test_percentage, validation_percentage):
    np.random.seed(6)
    data_dict = simulate_non_explainable_discrimination_admission_data(n)
    data_dict, discriminated_indices = add_discrimination_admission(data_dict, n_biased)

    n_samples_test = round(n * test_percentage)
    n_samples_val = round(n * validation_percentage)

    train_data_dict, test_data_dict, validation_data_dict = split_data_dict(data_dict, discriminated_indices, n_samples_test, n_samples_val)
    return train_data_dict, test_data_dict, validation_data_dict


def simulate_explainable_discrimination(n, n_biased, test_percentage, validation_percentage):
    np.random.seed(6)
    data_dict = simulate_explainable_discrimination_admission_data(n)
    data_dict, discriminated_indices = add_discrimination_admission(data_dict, n_biased)

    n_samples_test = round(n * test_percentage)
    n_samples_val = round(n * validation_percentage)

    train_data_dict, test_data_dict, validation_data_dict = split_data_dict(data_dict, discriminated_indices,
                                                                            n_samples_test, n_samples_val)
    return train_data_dict, test_data_dict, validation_data_dict