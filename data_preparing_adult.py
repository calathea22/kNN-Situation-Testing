import pandas as pd
import numpy as np

def load_adult_dataset(n):
    raw_data = pd.read_csv('adult.csv')
    raw_data = raw_data.replace({'?': np.nan}).dropna()
    sampled_data = raw_data.groupby('sex', group_keys=False).apply(lambda x: x.sample(n, random_state = 6))
    sampled_data.reset_index(drop=True, inplace=True)

    protected_var = 'sex'
    class_var = 'income'
    columns_to_drop = ['fnlwgt', 'education', 'occupation', 'race', 'relationship']

    class_info = sampled_data[class_var]
    class_labels = np.array([0 if x == '<=50K' else 1 for x in class_info])

    protected_info = sampled_data[protected_var]
    protected_labels = np.array([1 if x == 'Female' else 2 for x in protected_info])

    training_data = sampled_data.copy()
    #if private = 0
    training_data.loc[((training_data['workclass'] == 'Private')),'workclass'] = 0
    #if self_employed = 1
    training_data.loc[((training_data['workclass'] == 'Self-emp-not-inc')
                       | (training_data['workclass'] == 'Self-emp-inc')), 'workclass'] = 1
    # if government = 2 (never worked and without pay included, because they barely occur)
    training_data.loc[((training_data['workclass'] == 'Federal-gov')
                       | (training_data['workclass'] == 'Local-gov')
                       | (training_data['workclass'] == 'State-gov')
                       | (training_data['workclass'] == 'Without-pay')
                       | (training_data['workclass'] == 'Never-worked')), 'workclass'] = 2


    training_data.loc[training_data['native.country'] != 'United-States', 'native.country'] = 1
    training_data.loc[training_data['native.country'] == 'United-States', 'native.country'] = 0

    # if in some way married = 0
    training_data.loc[((training_data['marital.status'] == 'Married-AF-spouse') | (
                training_data['marital.status'] == 'Married-civ-spouse') | (
                training_data['marital.status'] == 'Married-spouse-absent')),'marital.status'] = 0

    #if widowed or never married = 1
    training_data.loc[((training_data['marital.status'] == 'Widowed') |
                       (training_data['marital.status'] == 'Never-married')), 'marital.status'] = 1

    #if divorced or seperated = 2
    training_data.loc[(training_data['marital.status'] == 'Divorced') |
                      (training_data['marital.status'] == 'Separated'), 'marital.status'] = 2

    training_data = pd.get_dummies(training_data, columns=['native.country', 'marital.status', 'workclass'])

    training_data = training_data.drop(columns=class_var)
    training_data = training_data.drop(columns=columns_to_drop)
    # We remove the protected attribute from the training data
    training_data = training_data.drop(columns=protected_var)

    return {'class_label': class_labels, 'protected_info': protected_labels, 'training_data': training_data}


def non_discriminated_adult_data_get_columns_info():
    interval_vars = ['age', 'capital.loss', 'hours.per.week', 'capital.gain']
    nominal_vars = []
    ordinal_vars = ['education.num', 'native.country_0', 'native.country_1', 'marital.status_0', 'marital.status_1', 'marital.status_2',
       'workclass_0', 'workclass_1', 'workclass_2']

    columns_info = {'interval': interval_vars, 'nominal': nominal_vars, 'ordinal': ordinal_vars}
    return columns_info


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

    train_data_dict = {'data': train_data, 'protected_info': protected_info_train, 'class_label': labels_train,
                       'indices_info': indices_info}
    test_data_dict = {'data': test_data, 'protected_info': protected_info_test, 'class_label': labels_test,
                      'indices_info': indices_info}
    validation_data_dict = {'data': validation_data, 'protected_info': protected_info_validation,
                            'class_label': labels_validation, 'indices_info': indices_info}

    return train_data_dict, test_data_dict, validation_data_dict

