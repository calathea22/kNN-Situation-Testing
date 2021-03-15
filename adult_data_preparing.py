import pandas as pd
import numpy as np
from data_preparing_utils import learn_classifier, get_doubtful_cases, split_discriminated_indices


def load_adult_dataset(n):
    raw_data = pd.read_csv('adult.csv')
    raw_data = raw_data.replace({'?': np.nan}).dropna()
    sampled_data = raw_data.groupby('sex', group_keys=False).apply(lambda x: x.sample(n, random_state = 6))
    sampled_data.reset_index(drop=True, inplace=True)

    protected_var = 'sex'
    class_var = 'income'
    columns_to_drop = ['fnlwgt', 'education']

    class_info = sampled_data[class_var]
    class_labels = np.array([0 if x == '<=50K' else 1 for x in class_info])

    protected_info = sampled_data[protected_var]
    protected_labels = np.array([1 if x == 'Female' else 2 for x in protected_info])

    training_data = sampled_data.drop(columns=class_var)
    training_data = training_data.drop(columns=columns_to_drop)
    # We remove the protected attribute from the training data
    training_data = training_data.drop(columns=protected_var)
    print(class_labels)
    protected_indices = np.where(protected_labels==1)
    print(protected_indices)
    unprotected_indices = np.where(protected_labels==2)
    print(sum(class_labels[protected_indices]))
    print(sum(class_labels[unprotected_indices]))
    print(training_data)

    return {'class_label': class_labels, 'protected_info': protected_labels, 'training_data': training_data}


def non_discriminated_adult_data_get_columns_info():
    interval_vars = ['age', 'capital.loss', 'hours.per.week', 'capital.gain']
    nominal_vars = ['race', 'marital.status', 'relationship', 'occupation', 'workclass']
    ordinal_vars = ['education.num']

    columns_info = {'interval': interval_vars, 'nominal': nominal_vars, 'ordinal': ordinal_vars}
    return columns_info


def discriminated_adult_data_get_columns_info():
    interval_vars = ['age', 'capital.loss', 'hours.per.week', 'capital.gain']
    nominal_vars = ['race', 'marital.status', 'relationship']
    ordinal_vars = ['education.num']

    columns_info = {'interval': interval_vars, 'nominal': nominal_vars, 'ordinal': ordinal_vars}
    return columns_info


def add_discrimination_adult(data_dict, biased_percentage):
    training_data = data_dict['training_data']
    gender = data_dict['protected_info']
    training_data['gender'] = gender
    org_class_labels = data_dict['class_label']

    protected_class_labels = org_class_labels[np.where(gender==1)[0]]
    unprotected_class_labels = org_class_labels[np.where(gender==2)[0]]
    number_of_positive_for_protected = sum(protected_class_labels)

    n_biased = int(number_of_positive_for_protected * biased_percentage)
    print(n_biased)

    attributes_to_find_randgevallen = ['gender', 'occupation', 'workclass']

    predicted_probabilites = learn_classifier(training_data[attributes_to_find_randgevallen], org_class_labels)
    print(predicted_probabilites)

    protected_indices = list(np.where(gender == 1)[0])
    predicted_probabilites_women = pd.DataFrame(predicted_probabilites[protected_indices])
    predicted_probabilites_women['Index'] = protected_indices

    discriminated_indices = get_doubtful_cases(predicted_probabilites_women, n_biased, org_class_labels)
    new_class_labels = org_class_labels.copy()
    np.put(new_class_labels, discriminated_indices, False)

    training_data = training_data.drop(columns=attributes_to_find_randgevallen)

    return ({'class_label': new_class_labels, 'disc_free_labels': org_class_labels, 'protected_info': gender, 'training_data': training_data}, discriminated_indices)


def load_adult_train_val_test_data(n, test_percentage, validation_percentage):
    np.random.seed(6)
    data_dict = load_adult_dataset(int(n/2))
    complete_data = data_dict['training_data']
    complete_protected_info = data_dict['protected_info']
    complete_labels = data_dict['class_label']

    n_samples_test = round(n * test_percentage)
    n_samples_val = round(n * validation_percentage)

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
    test_and_validation_indices = np.append(validation_data_indices, test_data_indices)
    protected_info_train = np.delete(complete_protected_info, test_and_validation_indices)
    labels_train = np.delete(complete_labels, test_and_validation_indices)

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    validation_data.reset_index(drop=True, inplace=True)

    columns_info = non_discriminated_adult_data_get_columns_info()
    interval_indices = [complete_data.columns.get_loc(var) for var in columns_info['interval']]
    nominal_indices = [complete_data.columns.get_loc(var) for var in columns_info['nominal']]
    ordinal_indices = [complete_data.columns.get_loc(var) for var in columns_info['ordinal']]
    indices_info = {'interval': interval_indices, 'nominal': nominal_indices, 'ordinal': ordinal_indices}
    print(indices_info)

    train_data_dict = {'data': train_data, 'protected_info': protected_info_train, 'class_label': labels_train,
                       'indices_info': indices_info}
    test_data_dict = {'data': test_data, 'protected_info': protected_info_test, 'class_label': labels_test,
                      'indices_info': indices_info}
    validation_data_dict = {'data': validation_data, 'protected_info': protected_info_validation,
                            'class_label': labels_validation, 'indices_info': indices_info}

    return train_data_dict, test_data_dict, validation_data_dict


def load_discriminated_adult_train_val_test_data(n, biased_percentage, test_percentage, validation_percentage):
    np.random.seed(6)
    data_dict = load_adult_dataset(int(n/2))

    data_dict, discriminated_indices = add_discrimination_adult(data_dict, biased_percentage)

    n_samples_test = round(n*test_percentage)
    n_samples_val = round(n*validation_percentage)

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
    train_discriminated_indices, test_discriminated_indices, validation_discriminated_indices = split_discriminated_indices(test_data_indices, train_data_indices, validation_data_indices, discriminated_indices)

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    validation_data.reset_index(drop=True, inplace=True)

    columns_info = discriminated_adult_data_get_columns_info()
    interval_indices = [complete_data.columns.get_loc(var) for var in columns_info['interval']]
    nominal_indices = [complete_data.columns.get_loc(var) for var in columns_info['nominal']]
    ordinal_indices = [complete_data.columns.get_loc(var) for var in columns_info['ordinal']]
    indices_info = {'interval': interval_indices, 'nominal': nominal_indices, 'ordinal': ordinal_indices}

    train_data_dict = {'data': train_data, 'protected_info': protected_info_train, 'class_label': labels_train, 'discriminated': train_discriminated_indices, 'indices_info': indices_info, 'disc_free_labels': disc_free_labels_train}
    test_data_dict = {'data': test_data, 'protected_info': protected_info_test, 'class_label': labels_test, 'discriminated': test_discriminated_indices, 'indices_info': indices_info, 'disc_free_labels': disc_free_labels_test}
    validation_data_dict = {'data': validation_data, 'protected_info': protected_info_validation, 'class_label': labels_validation,
                      'discriminated': validation_discriminated_indices, 'indices_info': indices_info, 'disc_free_labels': disc_free_labels_validation}

    return train_data_dict, test_data_dict, validation_data_dict

