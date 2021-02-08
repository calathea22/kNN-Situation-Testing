import utils
from load_data import load_data, split_test_sets
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder




def test_splitted_test_sets_disc_removal(location, n_splits):
    test_sets = split_test_sets(n_splits, location)
    results = []
    for test_set in test_sets:
        test_data = test_set['data']
        test_data_labels = test_set['class_label']
        train_without_discrimination_f1, without_massaging_f1, baseline_f1, luong_f1, zhang_f1, euclidean_f1, mahalanobis_f1 = test_discrimination_removal(
            disc_free_train_info, test_data, test_data_labels)
        results.append({'train_without_discrimination': train_without_discrimination_f1, 'without_massaging': without_massaging_f1, 'baseline': baseline_f1, 'luong': luong_f1, 'zhang': zhang_f1, 'euclidean': euclidean_f1,
                        'mahalanobis': mahalanobis_f1})
    print(results)
    get_avg_value_splitted_test_sets_disc_removal(results)
    return results

def get_test_predictions(X_train, y_train, X_test):
    enc = OneHotEncoder(handle_unknown='ignore')

    enc.fit(X_train)
    X_train_one_hot_encoded = enc.transform(X_train).toarray()
    X_test_one_hot_encoded = enc.transform(X_test).toarray()

    random_forest = RandomForestClassifier(random_state=100)
    random_forest.fit(X_train_one_hot_encoded, y_train)

    y_test_predicted = random_forest.predict(X_test_one_hot_encoded)
    return y_test_predicted


def test_discrimination_removal(disc_removal_info, test_data, test_data_labels):
    train_data = disc_removal_info['train_data']
    protected_info_train = disc_removal_info['protected_info_train']
    discriminated_train_labels = disc_removal_info['class_label_train']
    non_discriminated_train_labels = disc_removal_info['no_disc_class_label_train']
    train_data['Gender'] = protected_info_train

    train_label_baseline = disc_removal_info['train_labels_baseline']
    train_label_luong = disc_removal_info['train_labels_luong']
    train_label_zhang = disc_removal_info['train_labels_zhang']
    train_label_euclidean = disc_removal_info['train_labels_euclidean']
    train_label_mahalanobis = disc_removal_info['train_labels_mahalanobis']

    train_without_discrimination = get_test_predictions(train_data, non_discriminated_train_labels, test_data)
    without_massaging_labels = get_test_predictions(train_data, discriminated_train_labels, test_data)

    baseline_massaging_labels = get_test_predictions(train_data, train_label_baseline, test_data)
    luong_massaging_labels = get_test_predictions(train_data, train_label_luong, test_data)
    zhang_massaging_labels = get_test_predictions(train_data, train_label_zhang, test_data)
    euclidean_massaging_labels = get_test_predictions(train_data, train_label_euclidean, test_data)
    mahalanobis_massaging_labels = get_test_predictions(train_data, train_label_mahalanobis, test_data)

    train_without_discrimination_f1 = f1_score(test_data_labels, train_without_discrimination)
    without_massaging_f1 = f1_score(test_data_labels, without_massaging_labels)

    baseline_f1 = f1_score(test_data_labels, baseline_massaging_labels)
    luong_f1 = f1_score(test_data_labels, luong_massaging_labels)
    zhang_f1 = f1_score(test_data_labels, zhang_massaging_labels)
    euclidean_f1 = f1_score(test_data_labels, euclidean_massaging_labels)
    mahalanobis_f1 = f1_score(test_data_labels, mahalanobis_massaging_labels)

    return train_without_discrimination_f1, without_massaging_f1, baseline_f1, luong_f1, zhang_f1, euclidean_f1, mahalanobis_f1
