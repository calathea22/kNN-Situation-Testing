import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import PC, BayesianEstimator
import itertools
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score
import numpy as np



# Learn model from data
def make_model_admission(data):
    c = PC(data)
    model = c.estimate(significance_level=0.05)
    bayesian_model = BayesianModel(model.edges)
    print(bayesian_model.edges)
    bayesian_model = BayesianModel([('Extra Curricular', 'Admission'), ('Gender', 'Extra Curricular'), ('Score', 'Admission'), ('Gender', 'Admission'), ('Gender', 'A')])
    return bayesian_model


def make_model_adult(data):
    # c = PC(data)
    # model = c.estimate(significance_level=0.05)
    # bayesian_model = BayesianModel(model.edges)
    # print(bayesian_model.edges)
    edges = [('Gender', 'Income'), ('Gender', 'marital.status_0'), ('marital.status_0', 'Income'), ('Gender', 'workclass_1'), ('education.num', 'Income'), ('age', 'marital.status_1'), ('capital.gain', 'Income'), ('capital.gain', 'hours.per.week'), ('capital.loss', 'Income'), ('age', 'workclass_0'), ('age', 'hours.per.week'), ('native.country_1', 'education.num'), ('native.country_0', 'education.num')]
    bayesian_model = BayesianModel(edges)
    return bayesian_model


# Make different value assignments for Q
def find_Q_value_assignments(model, sens_attribute, decision_attribute, data):
    Q = model.get_parents(decision_attribute)
    Q.remove(sens_attribute)

    value_assignment_options = {}
    for parent in Q:
        unique_values = data[parent].unique()
        value_assignment_options[parent] = unique_values

    keys = value_assignment_options.keys()
    values = (value_assignment_options[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return Q, combinations


# given a list of possible value assignments for Q, find the data instances that match the value assignments
def add_instances_to_value_assignments(value_assignments, data):
    instances_per_value_assignment = []
    for value_assignment in value_assignments:
        temp_data = data
        for key, value in value_assignment.items():
            temp_data = temp_data[temp_data[key] == value]

        instances_per_value_assignment.append(temp_data)
    return instances_per_value_assignment


def get_conditional_probability(cpd_table, evidence_values):
    temp = cpd_table.values

    order_of_keys = cpd_table.state_names.copy()

    for evidence_key in order_of_keys.keys():
        evidence_value = evidence_values[evidence_key]
        indeces = order_of_keys[evidence_key]
        if (evidence_value not in indeces):
            return 0
        evidence_index = indeces.index(evidence_value)
        temp = temp[evidence_index]
    return temp



# q is a dict, q_k and q_k_apostroph are tubles of the form ('attribute', attribute_value)
def causal_effect(q, q_k_apostroph, cpd_decision, cpd_sens, sens_attribute_name, dec_attribute_name):
    cpd_sens_probabilities = cpd_sens.values
    sens_attribute_values = cpd_sens.state_names[sens_attribute_name]
    causal_effect = 0
    q_k_apostroph_plus_q = dict(q)
    q_k_apostroph_plus_q[q_k_apostroph[0]] = q_k_apostroph[1]

    for sens_attribute_value in sens_attribute_values:
        q[dec_attribute_name] = 1
        q[sens_attribute_name] = sens_attribute_value

        q_k_apostroph_plus_q[dec_attribute_name] =1
        q_k_apostroph_plus_q[sens_attribute_name] = sens_attribute_value

        prob_sens_att = cpd_sens_probabilities[sens_attribute_values.index(sens_attribute_value)]
        first_part_equation = get_conditional_probability(cpd_decision, q) * prob_sens_att
        second_part_equation = get_conditional_probability(cpd_decision, q_k_apostroph_plus_q) * prob_sens_att
        causal_effect += (first_part_equation-second_part_equation)

    return abs(causal_effect)


def value_difference(q_k, q_k_apostroph, indices_info, range_dict):
    interval_columns = indices_info['interval']
    ordinal_columns = indices_info['ordinal']
    attribute = q_k[0]

    if (attribute in interval_columns) or (attribute in ordinal_columns):
        return abs(q_k[1]-q_k_apostroph[1])/range_dict[attribute]
    else:
        return q_k[1] != q_k_apostroph[1]


# distance formula
def define_distance(tuple_in_question, neighbour_tuple, Q, cpd_decision, cpd_sens, indices_info, range_dict, sens_attribute_name, dec_attribute_name):
    distance = 0
    tuple_in_question_dict = tuple_in_question[Q].to_dict()
    for parent in Q:
        q_k = (parent, tuple_in_question[parent])
        q_k_apostroph = (parent, neighbour_tuple[parent])
        distance += value_difference(q_k, q_k_apostroph, indices_info, range_dict) * causal_effect(tuple_in_question_dict, q_k_apostroph, cpd_decision, cpd_sens, sens_attribute_name, dec_attribute_name)
    return distance


def define_distances(Q, tuple_in_question, value_assignments, instances_for_value_assignments, cpd_decision, cpd_sens, indices_info, range_dict,  sens_attribute_name, dec_attribute_name):
# probability work with some dictionary, where store all the tuples and the distance between tuples, to tuple in question
    distances = []
    for value_assignment, instances_for_value_assignment in zip(value_assignments, instances_for_value_assignments):
        if (len(instances_for_value_assignment) != 0):
            distances.append(
                define_distance(tuple_in_question, instances_for_value_assignment.iloc[0], Q, cpd_decision, cpd_sens,
                                indices_info, range_dict, sens_attribute_name, dec_attribute_name))
        else:
            distances.append(100000)
    return distances


def order_distances(instances_per_value_assignment, distances):
    zipped = zip(distances, instances_per_value_assignment)
    ordered_value_assignments = [x for _, x in sorted(zipped, key=lambda t: t[0])]
    return ordered_value_assignments

#here we'll implement the while loop selecting n tuples from the ordered distances
def disc_score_one_instance(k, ordered_value_assignments, sens_attribute, dec_attribute, indices_info):
    number_of_neighbours = 0
    index = 0
    protected_nearest_neighbours_labels = []
    unprotected_nearest_neighbours_lables = []
    while(number_of_neighbours < k):
        instances = ordered_value_assignments[index]
        protected_neighbours = instances[instances[sens_attribute] == 1]
        unprotected_neighbours = instances[instances[sens_attribute] == 2]
        neighbours_to_be_selected = min([len(protected_neighbours), len(unprotected_neighbours), k-number_of_neighbours])

        protected_nearest_neighbours_labels.extend(protected_neighbours[dec_attribute].iloc[0:neighbours_to_be_selected].values)
        unprotected_nearest_neighbours_lables.extend(unprotected_neighbours[dec_attribute].iloc[0:neighbours_to_be_selected].values)

        number_of_neighbours += neighbours_to_be_selected
        index += 1

    p1 = sum(unprotected_nearest_neighbours_lables)/k
    p2 = sum(protected_nearest_neighbours_labels)/k

    return p1-p2


def give_info_about_selected_neighbours(selected_neighbours_indeces, indices_info, org_train_data):
    non_binned_neighbours = org_train_data.iloc[selected_neighbours_indeces]
    for interval_column in indices_info['interval']:
        print(interval_column)
        print("Mean value of neighbours for this feature: " + str(non_binned_neighbours[interval_column].mean()))
    for ordinal_column in indices_info['ordinal']:
        print(ordinal_column)
        print(non_binned_neighbours.groupby(ordinal_column).count())


def disc_score_one_instance_1k(k, ordered_value_assignments, sens_attribute, dec_attribute, indices_info, org_train_data):
    number_of_neighbours = 0
    index = 0
    unprotected_nearest_neighbours_lables = []
    indices_selected_neighbours = []
    while(number_of_neighbours < k):
        instances = ordered_value_assignments[index]
        unprotected_neighbours = instances[instances[sens_attribute] == 2]
        neighbours_to_be_selected = min(k-number_of_neighbours, len(unprotected_neighbours))

        unprotected_nearest_neighbours_lables.extend(unprotected_neighbours[dec_attribute].iloc[0:neighbours_to_be_selected].values)
        selected_neighbours = unprotected_neighbours.iloc[0:neighbours_to_be_selected]
        indices_selected_neighbours.extend(selected_neighbours.index)
        number_of_neighbours += neighbours_to_be_selected
        index += 1

    #give_info_about_selected_neighbours(indices_selected_neighbours, indices_info, org_train_data)
    p1 = sum(unprotected_nearest_neighbours_lables)/k
    # p2 = sum(protected_nearest_neighbours_labels)/k
    return p1


def decision_label_one_instance_unprotected_group(k, ordered_value_assignments, sens_attribute, dec_attribute):
    number_of_neighbours = 0
    index = 0
    unprotected_nearest_neighbours_lables = []
    while(number_of_neighbours < k):
        instances = ordered_value_assignments[index]
        # protected_neighbours = instances[instances[sens_attribute] == 1]
        unprotected_neighbours = instances[instances[sens_attribute] == 2]
        amount_of_selected_neighbours = min(k-number_of_neighbours, len(unprotected_neighbours))
        unprotected_nearest_neighbours_lables.extend(unprotected_neighbours[dec_attribute].iloc[0:amount_of_selected_neighbours].values)
        number_of_neighbours += amount_of_selected_neighbours
        index += 1
    ratio_of_positive_class_labels_neighbours = sum(unprotected_nearest_neighbours_lables)/k
    ratio_of_negative_class_labels_neighbours = 1 - ratio_of_positive_class_labels_neighbours
    decision_score = (ratio_of_positive_class_labels_neighbours - ratio_of_negative_class_labels_neighbours)
    return decision_score


def sensitive_label_one_instance(k, ordered_value_assignments, sens_attribute, dec_attribute):
    number_of_neighbours = 0
    index = 0
    nearest_neighbours_gender_lables = []
    while(number_of_neighbours < k):
        instances = ordered_value_assignments[index]
        amount_of_selected_neighbours = min(k-number_of_neighbours, len(instances))
        nearest_neighbours_gender_lables.extend(instances[sens_attribute].sample(amount_of_selected_neighbours))
        # nearest_neighbours_gender_lables.extend(instances[sens_attribute].iloc[0:amount_of_selected_neighbours].values)
        number_of_neighbours += amount_of_selected_neighbours
        index += 1
    nearest_neighbours_gender_lables = np.array(nearest_neighbours_gender_lables)
    ratio_of_unprotected_neighbours = len(nearest_neighbours_gender_lables[np.where(nearest_neighbours_gender_lables==2)])/k
    # print(ratio_of_protected_neighbours)
    if (ratio_of_unprotected_neighbours >= 0.5):
        return 2
    return 1



def give_discrimination_scores_zhang(model, sens_attribute, decision_attribute, train_data, test_data, k, indices_info, range_dict, protected_indices, org_train_data=[]):
    model.fit(train_data, estimator=BayesianEstimator)
    cpd_decision = model.get_cpds(decision_attribute)
    cpd_sens = model.get_cpds(sens_attribute)
    Q, value_assignments = find_Q_value_assignments(model, sens_attribute, decision_attribute, train_data)
    instances_per_value_assignment = add_instances_to_value_assignments(value_assignments, train_data)

    protected_data_test = test_data.iloc[protected_indices]

    disc_scores = []
    for i in range(len(protected_data_test)):
        if protected_data_test[decision_attribute].iloc[i] == 1:
            disc_scores.append(-1.0)
        else:
            distances = define_distances(Q, protected_data_test.iloc[i], value_assignments, instances_per_value_assignment, cpd_decision,
                                         cpd_sens, indices_info, range_dict, sens_attribute, decision_attribute)
            ordered_value_assignments = order_distances(instances_per_value_assignment, distances)
            disc_score = disc_score_one_instance_1k(k, ordered_value_assignments, sens_attribute, decision_attribute, indices_info, org_train_data)
            disc_scores.append(disc_score)
    return disc_scores


def give_decision_labels_zhang_unprotected_group(model, sens_attribute, decision_attribute, train_data, test_data, k, indices_info, range_dict, protected_label, unprotected_label):
    model.fit(train_data, estimator=BayesianEstimator)
    cpd_decision = model.get_cpds(decision_attribute)
    cpd_sens = model.get_cpds(sens_attribute)
    Q, value_assignments = find_Q_value_assignments(model, sens_attribute, decision_attribute, train_data)
    instances_per_value_assignment = add_instances_to_value_assignments(value_assignments, train_data)

    unprotected_indices_test_set = test_data[sens_attribute] == unprotected_label
    unprotected_data_test = test_data[unprotected_indices_test_set]

    predictions_negative_class = []
    predictions_positive_class = []
    all_predictions = []
    actual_decision_labels = []
    for i in range(len(unprotected_data_test)):
        unprotected_test_instance = unprotected_data_test.iloc[i]
        distances = define_distances(Q, unprotected_test_instance, value_assignments, instances_per_value_assignment, cpd_decision,
                                     cpd_sens, indices_info, range_dict, sens_attribute, decision_attribute)
        ordered_value_assignments = order_distances(instances_per_value_assignment, distances)
        decision_score = decision_label_one_instance_unprotected_group(k, ordered_value_assignments, sens_attribute, decision_attribute)
        #als het goed is zit in unprotected_test_instance het decision label
        actual_decision_label = unprotected_test_instance[decision_attribute]
        all_predictions.append(decision_score>=0.5)
        actual_decision_labels.append(actual_decision_label)
        if (actual_decision_label == 1):
            predictions_positive_class.append(decision_score)
        else:
            predictions_negative_class.append(decision_score)
    print(accuracy_score(actual_decision_labels, all_predictions))
    return predictions_negative_class, predictions_positive_class



def predict_sens_attribute_zhang(model, sens_attribute, decision_attribute, train_data, test_data, k, indices_info, range_dict, protected_label, unprotected_label):
    model.fit(train_data, estimator=BayesianEstimator)
    cpd_decision = model.get_cpds(decision_attribute)
    cpd_sens = model.get_cpds(sens_attribute)
    Q, value_assignments = find_Q_value_assignments(model, sens_attribute, decision_attribute, train_data)
    instances_per_value_assignment = add_instances_to_value_assignments(value_assignments, train_data)

    all_predictions = []
    actual_gender_labels = []
    for i in range(len(test_data)):
        test_instance = test_data.iloc[i]
        distances = define_distances(Q, test_instance, value_assignments, instances_per_value_assignment, cpd_decision,
                                     cpd_sens, indices_info, range_dict, sens_attribute, decision_attribute)
        ordered_value_assignments = order_distances(instances_per_value_assignment, distances)
        gender_label = sensitive_label_one_instance(k, ordered_value_assignments, sens_attribute, decision_attribute)
        #als het goed is zit in unprotected_test_instance het decision label
        actual_gender = test_instance[sens_attribute]
        all_predictions.append(gender_label)
        actual_gender_labels.append(actual_gender)


    print(accuracy_score(actual_gender_labels, all_predictions))
    return



def give_discrimination_labels_zhang(discrimination_scores, treshold):
    disc_labels = []
    for score in discrimination_scores:
        disc_labels.append(score > treshold)
    return disc_labels


def define_distance_for_matrix(tuple_in_question, neighbour_tuple, Q_index, Q, cpd_decision, cpd_sens, indices_info, range_dict, sens_attribute_name, dec_attribute_name):
    distance = 0

    tuple_in_question = pd.Series(tuple_in_question[Q_index], index=Q)
    neighbour_tuple = pd.Series(neighbour_tuple[Q_index], index=Q)

    tuple_in_question_dict = tuple_in_question[Q].to_dict()
    for parent in Q:
        q_k = (parent, tuple_in_question[parent])
        q_k_apostroph = (parent, neighbour_tuple[parent])
        distance += value_difference(q_k, q_k_apostroph, indices_info, range_dict) * causal_effect(
            tuple_in_question_dict, q_k_apostroph, cpd_decision, cpd_sens, sens_attribute_name, dec_attribute_name)
    return distance


def make_distance_matrix_zhang(data, model, sens_attribute, decision_attribute, indices_info, range_dict):
    model.fit(data, estimator=BayesianEstimator)
    Q, value_assignments = find_Q_value_assignments(model, sens_attribute, decision_attribute, data)
    Q_indices = [data.columns.get_loc(q) for q in Q]
    cpd_decision = model.get_cpds(decision_attribute)
    cpd_sens = model.get_cpds(sens_attribute)
    dists = pdist(data, define_distance_for_matrix, Q_index=Q_indices, Q=Q, cpd_decision=cpd_decision, cpd_sens=cpd_sens, indices_info=indices_info,
                  range_dict=range_dict, sens_attribute_name=sens_attribute, dec_attribute_name=decision_attribute)
    distance_matrix = pd.DataFrame(squareform(dists))
    return distance_matrix
