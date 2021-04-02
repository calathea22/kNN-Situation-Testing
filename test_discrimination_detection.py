from load_data import load_data, load_optimization_info, split_test_sets

import utils
from sklearn.metrics import f1_score
from giving_discrimination_scores import give_all_disc_scores_k_approach


def area_under_curve_all_approaches(loaded_train_data, loaded_test_data, loaded_optimization_info, indices_info, k_info, adult_or_admission):
    val_ground_truth = loaded_test_data['ground_truth']

    baseline_scores, luong_scores, zhang_scores, weighted_euclidean_scores, mahalanobis_scores = give_all_disc_scores_k_approach(loaded_train_data, loaded_test_data, loaded_optimization_info, indices_info, k_info, adult_or_admission)
    print(sum(val_ground_truth))

    print("Baseline")
    auc_baseline_pr, auc_baseline_roc = utils.get_auc_scores(val_ground_truth, baseline_scores)
    print("Luong")
    auc_luong_pr, auc_luong_roc = utils.get_auc_scores(val_ground_truth, luong_scores)
    print("Zhang")
    auc_zhang_pr, auc_zhang_roc = utils.get_auc_scores(val_ground_truth, zhang_scores)
    print("Euclidean")
    auc_euclidean_pr, auc_euclidean_roc = utils.get_auc_scores(val_ground_truth, weighted_euclidean_scores)
    print("Mahalanobis")
    auc_mahalanobis_pr, auc_mahalanobis_roc = utils.get_auc_scores(val_ground_truth, mahalanobis_scores)

    return {'baseline': auc_baseline_pr, 'luong':auc_luong_pr, 'zhang': auc_zhang_pr, 'euclidean': auc_euclidean_pr, 'mahalanobis': auc_mahalanobis_pr}, \
           {'baseline': auc_baseline_roc, 'luong': auc_luong_roc, 'zhang': auc_zhang_roc, 'euclidean': auc_euclidean_roc, 'mahalanobis': auc_mahalanobis_roc}


def f1_score_all_approaches(loaded_train_data, loaded_test_data, loaded_optimization_info, indices_info, k_info, threshold_info, adult_or_admission="admission"):
    val_ground_truth = loaded_test_data['ground_truth']

    baseline_scores, luong_scores, zhang_scores, weighted_euclidean_scores, mahalanobis_scores = give_all_disc_scores_k_approach(loaded_train_data, loaded_test_data,
                                                                                                                                 loaded_optimization_info, indices_info, k_info, adult_or_admission)

    disc_labels_baseline = utils.give_disc_label(baseline_scores, threshold_info['baseline'])
    disc_labels_luong = utils.give_disc_label(luong_scores, threshold_info['luong'])
    disc_labels_zhang = utils.give_disc_label(zhang_scores, threshold_info['zhang'])
    disc_labels_euclidean = utils.give_disc_label(weighted_euclidean_scores, threshold_info['euclidean'])
    disc_labels_mahalanobis = utils.give_disc_label(mahalanobis_scores, threshold_info['mahalanobis'])

    f1_score_baseline = f1_score(val_ground_truth, disc_labels_baseline)
    f1_score_luong = f1_score(val_ground_truth, disc_labels_luong)
    f1_score_zhang = f1_score(val_ground_truth, disc_labels_zhang)
    f1_score_euclidean = f1_score(val_ground_truth, disc_labels_euclidean)
    f1_score_mahalanobis = f1_score(val_ground_truth, disc_labels_mahalanobis)

    return f1_score_baseline, f1_score_luong, f1_score_zhang, f1_score_euclidean, f1_score_mahalanobis


def area_under_curve_validation_set(location, indices_info, k_info, best_lambda_euclidean, best_lambda_mahalanobis):
    loaded_train_data = load_data(location, "train")
    loaded_val_data = load_data(location, "val")
    loaded_optimization_info = load_optimization_info(location, best_lambda_euclidean, best_lambda_mahalanobis)

    print(area_under_curve_all_approaches(loaded_train_data, loaded_val_data, loaded_optimization_info, indices_info, k_info))
    return


def area_under_curve_validation_set_different_k(location, indices_info, possible_ks, best_lambda_euclidean, best_lambda_mahalanobis):
    loaded_train_data = load_data(location, "train")
    loaded_val_data = load_data(location, "val")
    loaded_optimization_info = load_optimization_info(location, best_lambda_euclidean, best_lambda_mahalanobis)
    for k in possible_ks:
        print(k)
        k_info = {'baseline': k, 'luong': k, 'zhang':k, 'euclidean': k, 'mahalanobis': k}
        print(area_under_curve_all_approaches(loaded_train_data, loaded_val_data, loaded_optimization_info, indices_info, k_info))
    return


def area_under_curve_test_sets(location, indices_info, n_splits, k_info, best_lambda_euclidean, best_lambda_mahalanobis, adult_or_admission="admission"):
    loaded_train_data = load_data(location, "train")
    #loaded_optimization_info = {}
    loaded_optimization_info = load_optimization_info(location, best_lambda_euclidean, best_lambda_mahalanobis)
    splitted_test_sets = split_test_sets(n_splits, location)
    i = 0
    roc_results = []
    pr_results = []
    for test_set in splitted_test_sets:
        print("Split: " + str(i))
        pr_aucs, roc_aucs = area_under_curve_all_approaches(loaded_train_data, test_set, loaded_optimization_info, indices_info, k_info, adult_or_admission)
        roc_results.append({'baseline': roc_aucs['baseline'], 'luong': roc_aucs['luong'], 'zhang': roc_aucs['zhang'], 'euclidean': roc_aucs['euclidean'], 'mahalanobis': roc_aucs['mahalanobis']})
        pr_results.append({'baseline': pr_aucs['baseline'],'luong': pr_aucs['luong'], 'zhang': pr_aucs['zhang'], 'euclidean': pr_aucs['euclidean'], 'mahalanobis': pr_aucs['mahalanobis']})
        # roc_results.append({'baseline': roc_aucs['baseline'], 'luong': roc_aucs['luong'], 'zhang': roc_aucs['zhang'], 'euclidean': roc_aucs['euclidean'], 'mahalanobis': roc_aucs['mahalanobis']})
        # pr_results.append({'baseline': pr_aucs['baseline'], 'luong': pr_aucs['luong'], 'zhang': pr_aucs['zhang'], 'euclidean': pr_aucs['euclidean'], 'mahalanobis': pr_aucs['mahalanobis']})
        i += 1
        print(pr_aucs)
        print(roc_aucs)
    utils.print_avg_results_from_dictionary(roc_results)
    utils.print_avg_results_from_dictionary(pr_results)

    return


def f1_score_test_sets(location, indices_info, n_splits, k_info, threshold_info, best_lambda_euclidean, best_lambda_mahalanobis, adult_or_admission="admission"):
    loaded_train_data = load_data(location, "train")
    loaded_optimization_info = load_optimization_info(location, best_lambda_euclidean, best_lambda_mahalanobis)

    splitted_test_sets = split_test_sets(n_splits, location)
    i = 0
    f1_results = []

    for test_set in splitted_test_sets:
        print("Split: " + str(i))
        f1_baseline, f1_luong, f1_zhang, f1_euclidean, f1_mahalanobis = f1_score_all_approaches(
            loaded_train_data, test_set, loaded_optimization_info, indices_info, k_info, threshold_info, adult_or_admission)
        f1_results.append({'baseline': f1_baseline, 'luong': f1_luong, 'zhang': f1_zhang, 'euclidean': f1_euclidean, 'mahalanobis': f1_mahalanobis})
        print(f1_results)
        i += 1
    print("F1")
    utils.print_avg_results_from_dictionary(f1_results)


def f1_score_val_set_different_thresholds(location, indices_info, k_info, possible_thresholds, best_lambda_euclidean, best_lambda_mahalanobis, adult_or_admission="admission"):
    loaded_train_data = load_data(location, "train")
    loaded_val_data = load_data(location, "val")
    loaded_optimization_info = load_optimization_info(location, best_lambda_euclidean, best_lambda_mahalanobis)

    for threshold in possible_thresholds:
        print("threshold: " + str(threshold))
        threshold_info = {'baseline': threshold, 'luong': threshold, 'zhang': threshold, 'euclidean': threshold, 'mahalanobis': threshold}
        f1_baseline, f1_luong, f1_zhang, f1_euclidean, f1_mahalanobis = f1_score_all_approaches(loaded_train_data, loaded_val_data, loaded_optimization_info, indices_info, k_info, threshold_info, adult_or_admission)
        print(f1_baseline)
        print(f1_luong)
        print(f1_zhang)
        print(f1_euclidean)
        print(f1_mahalanobis)
    return

