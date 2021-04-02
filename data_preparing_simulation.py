import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from scipy import stats
import random
random.seed(6)
from data_preparing_utils import learn_classifier, get_doubtful_cases, split_discriminated_indices
import utils

def add_discrimination_admission(data_dict, biased_percentage):
    training_data = data_dict['training_data']

    classifying_data = training_data[['A', 'Score', 'Extra Curricular']].copy()
    gender = data_dict['protected_info']
    classifying_data['gender'] = gender
    org_class_labels = data_dict['class_label']
    number_of_women = sum(gender[np.where(gender==1)])
    protected_class_labels = org_class_labels[np.where(gender==1)]
    number_of_positive_for_protected = sum(protected_class_labels)
    n_biased = int(number_of_positive_for_protected * biased_percentage)
    #n_biased = number_of_women * biased_percentage
    print(n_biased)

    predicted_probabilites = learn_classifier(classifying_data, org_class_labels)

    protected_indices = list(np.where(gender == 1)[0])
    predicted_probabilites_women = pd.DataFrame(predicted_probabilites[protected_indices])
    predicted_probabilites_women['Index'] = protected_indices

    discriminated_indices = get_doubtful_cases(predicted_probabilites_women, n_biased, org_class_labels)
    new_class_labels = org_class_labels.copy()
    np.put(new_class_labels, discriminated_indices, False)

    return ({'class_label': new_class_labels, 'disc_free_labels': org_class_labels, 'protected_info': gender, 'training_data': training_data}, discriminated_indices)


def admission_data_get_columns_info():
    interval_vars = ['Height', 'Score', 'Extra Curricular']
    nominal_vars = []
    ordinal_vars = []

    columns_info = {'interval': interval_vars, 'nominal': nominal_vars, 'ordinal': ordinal_vars}
    return columns_info


def simulate_non_explainable_discrimination_admission_data(n):
    np.random.seed(6)
    gender = np.random.randint(2, size=n) + 1

    female_indices = np.where(gender==1)[0]
    male_indices = np.where(gender==2)[0]

    female_a = np.round(np.random.normal(170, 2, len(female_indices)))
    male_a = np.round(np.random.normal(175, 2, len(male_indices)))

    a = np.zeros(n)
    a.put(female_indices, female_a)
    a.put(male_indices, male_a)

    testscore = np.round(np.random.normal(5, 2, n))

    extra_curricular = np.round(np.random.normal(5, 2, n))

    admission = 3 * testscore + 3 * extra_curricular + np.random.normal(5, 2, n)
    admission_probabilities = (admission - min(admission)) / (max(admission) - min(admission))
    admission_labels = [admission_probability > 0.5 for admission_probability in admission_probabilities]

    dataframe = pd.DataFrame(list(zip(a, testscore, extra_curricular)),
                             columns=['A', 'Score', 'Extra Curricular'])

    return {'class_label': np.array(admission_labels), 'protected_info': np.array(gender), 'training_data': dataframe}


def simulate_non_explainable_discrimination_4_attributes_admission_data(n):
    np.random.seed(6)
    gender = np.random.randint(2, size=n) + 1

    female_indices = np.where(gender==1)[0]
    male_indices = np.where(gender==2)[0]

    female_a = np.round(np.random.normal(170, 2, len(female_indices)))
    male_a = np.round(np.random.normal(175, 2, len(male_indices)))
    a = np.zeros(n)
    a.put(female_indices, female_a)
    a.put(male_indices, male_a)

    testscore = np.round(np.random.normal(5, 2, n))

    extra_curricular = np.round(np.random.normal(5, 2, n))

    admission = 3 * testscore + 3 * extra_curricular + np.random.normal(5, 2, n)
    admission_probabilities = (admission - min(admission)) / (max(admission) - min(admission))
    admission_labels = [admission_probability > 0.5 for admission_probability in admission_probabilities]


    dataframe = pd.DataFrame(list(zip(a, a, a, a, testscore, extra_curricular)),
                             columns=['A', 'B', 'C', 'D', 'Score', 'Extra Curricular'])

    protected_attribute = np.array(gender)
    class_label = np.array(admission_labels)

    return {'class_label': np.array(admission_labels), 'protected_info': np.array(gender), 'training_data': dataframe}


def simulate_non_explainable_discrimination_7_attributes_admission_data(n):
    np.random.seed(6)
    gender = np.random.randint(2, size=n) + 1

    female_indices = np.where(gender == 1)[0]
    male_indices = np.where(gender == 2)[0]

    female_a = np.round(np.random.normal(170, 2, len(female_indices)))
    male_a = np.round(np.random.normal(175, 2, len(male_indices)))
    a = np.zeros(n)
    a.put(female_indices, female_a)
    a.put(male_indices, male_a)

    testscore = np.round(np.random.normal(5, 2, n))

    extra_curricular = np.round(np.random.normal(5, 2, n))

    admission = 3 * testscore + 3 * extra_curricular + np.random.normal(5, 2, n)
    admission_probabilities = (admission - min(admission)) / (max(admission) - min(admission))
    admission_labels = [admission_probability > 0.5 for admission_probability in admission_probabilities]

    dataframe = pd.DataFrame(list(zip(a, a, a, a, a, a, a, testscore, extra_curricular)),
                             columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Score', 'Extra Curricular'])

    protected_attribute = np.array(gender)
    class_label = np.array(admission_labels)

    return {'class_label': np.array(admission_labels), 'protected_info': np.array(gender), 'training_data': dataframe}


def simulate_explainable_discrimination_admission_data(n):
    np.random.seed(6)
    gender = np.random.randint(2, size=n) + 1

    female_indices = np.where(gender == 1)[0]
    male_indices = np.where(gender == 2)[0]

    female_height = np.round(np.random.normal(170, 2, len(female_indices)))
    male_height = np.round(np.random.normal(175, 2, len(male_indices)))

    height = np.zeros(n)
    height.put(female_indices, female_height)
    height.put(male_indices, male_height)

    testscore = np.round(np.random.normal(5, 2, n))
    extra_curricular = 3*gender + np.round(np.random.normal(5, 2, n))

    admission = 3 * testscore + 4 * extra_curricular + np.random.normal(5, 2, n)
    admission_probabilities = (admission - min(admission)) / (max(admission) - min(admission))
    admission_labels = [admission_probability > 0.5 for admission_probability in admission_probabilities]

    dataframe = pd.DataFrame(list(zip(height, testscore, extra_curricular)),
                             columns=['A', 'Score', 'Extra Curricular'])

    protected_attribute = np.array(gender)
    class_label = np.array(admission_labels)
    protected_class_labels = class_label[np.where(protected_attribute == 1)]
    print(sum(protected_class_labels))
    return {'class_label': np.array(admission_labels), 'protected_info': np.array(gender), 'training_data': dataframe}

def simulate_explainable_discrimination_4_attributes_admission_data(n):
    np.random.seed(6)
    gender = np.random.randint(2, size=n) + 1

    female_indices = np.where(gender==1)[0]
    male_indices = np.where(gender==2)[0]

    female_a = np.round(np.random.normal(170, 2, len(female_indices)))
    male_a = np.round(np.random.normal(175, 2, len(male_indices)))
    a = np.zeros(n)
    a.put(female_indices, female_a)
    a.put(male_indices, male_a)

    testscore = np.round(np.random.normal(5, 2, n))
    extra_curricular = 3*gender + np.round(np.random.normal(5, 2, n))
    print(extra_curricular)

    admission = 3 * testscore + 4 * extra_curricular + np.random.normal(5, 2, n)
    admission_probabilities = (admission - min(admission)) / (max(admission) - min(admission))
    admission_labels = [admission_probability > 0.5 for admission_probability in admission_probabilities]


    dataframe = pd.DataFrame(list(zip(a, a, a, a, testscore, extra_curricular)),
                             columns=['A', 'B', 'C', 'D', 'Score', 'Extra Curricular'])

    protected_attribute = np.array(gender)
    class_label = np.array(admission_labels)
    protected_class_labels = class_label[np.where(protected_attribute == 1)]
    print(sum(protected_class_labels))

    return {'class_label': np.array(admission_labels), 'protected_info': np.array(gender), 'training_data': dataframe}

def simulate_explainable_discrimination_7_attributes_admission_data(n):
    np.random.seed(6)
    gender = np.random.randint(2, size=n) + 1

    female_indices = np.where(gender==1)[0]
    male_indices = np.where(gender==2)[0]

    female_a = np.round(np.random.normal(170, 2, len(female_indices)))
    male_a = np.round(np.random.normal(175, 2, len(male_indices)))
    a = np.zeros(n)
    a.put(female_indices, female_a)
    a.put(male_indices, male_a)

    testscore = np.round(np.random.normal(5, 2, n))
    extra_curricular = 3*gender + np.round(np.random.normal(5, 2, n))

    admission = 3 * testscore + 4 * extra_curricular + np.random.normal(5, 2, n)
    admission_probabilities = (admission - min(admission)) / (max(admission) - min(admission))
    admission_labels = [admission_probability > 0.5 for admission_probability in admission_probabilities]

    dataframe = pd.DataFrame(list(zip(a, a, a, a, a, a, a, testscore, extra_curricular)),
                             columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Score', 'Extra Curricular'])


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
    complete_discrimination_free_labels = data_dict['disc_free_labels']

    test_data = complete_data.sample(n=n_samples_test)
    test_data_indices = test_data.index
    protected_info_test = complete_protected_info[test_data_indices]
    labels_test = complete_labels[test_data_indices]
    disc_free_labels_test = complete_discrimination_free_labels[test_data_indices]

    train_data = complete_data.drop(test_data_indices)
    validation_data = train_data.sample(n=n_samples_val)
    validation_data_indices = validation_data.index
    protected_info_validation = complete_protected_info[validation_data_indices]
    labels_validation = complete_labels[validation_data_indices]
    disc_free_labels_validation = complete_discrimination_free_labels[validation_data_indices]

    train_data = train_data.drop(validation_data_indices)
    train_data_indices = train_data.index
    test_and_validation_indices = np.append(validation_data_indices, test_data_indices)
    protected_info_train = np.delete(complete_protected_info, test_and_validation_indices)
    labels_train = np.delete(complete_labels, test_and_validation_indices)
    disc_free_labels_train = np.delete(complete_discrimination_free_labels, test_and_validation_indices)
    train_discriminated_indices, test_discriminated_indices, validation_discriminated_indices = split_discriminated_indices(
        test_data_indices, train_data_indices, validation_data_indices, discriminated_indices)

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    validation_data.reset_index(drop=True, inplace=True)

    train_data_dict = {'data': train_data, 'protected_info': np.array(protected_info_train), 'class_label': labels_train,
                       'discriminated': train_discriminated_indices, 'disc_free_labels': disc_free_labels_train}
    test_data_dict = {'data': test_data, 'protected_info': protected_info_test, 'class_label': labels_test,
                      'discriminated': test_discriminated_indices, 'disc_free_labels': disc_free_labels_test}
    validation_data_dict = {'data': validation_data, 'protected_info': protected_info_validation,
                            'class_label': labels_validation,
                            'discriminated': validation_discriminated_indices, 'disc_free_labels': disc_free_labels_validation}
    return train_data_dict, test_data_dict, validation_data_dict


def simulate_train_test_and_val_with_discrimination(n, biased_percentage, test_percentage, validation_percentage, simulate_function):
    np.random.seed(6)
    data_dict = simulate_function(n)

    data_dict, discriminated_indices = add_discrimination_admission(data_dict, biased_percentage)

    n_samples_test = round(n * test_percentage)
    n_samples_val = round(n * validation_percentage)

    train_data_dict, test_data_dict, validation_data_dict = split_data_dict(data_dict, discriminated_indices,
                                                                            n_samples_test, n_samples_val)
    return train_data_dict, test_data_dict, validation_data_dict
