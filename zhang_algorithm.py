import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import PC, BayesianEstimator
import itertools
from scipy.spatial.distance import pdist, squareform



# Learn model from data
def make_model_admission(data):
    c = PC(data)
    model = c.estimate(significance_level=0.05)
    bayesian_model = BayesianModel(model.edges)
    print(bayesian_model.edges)
    bayesian_model = BayesianModel([('Extra Curricular', 'Admission'), ('Score', 'Admission'), ('Gender', 'Admission'), ('Gender', 'Height')])
    return bayesian_model


def make_model_adult(data):
    # print(data)
    # c = PC(data)
    # model = c.estimate(significance_level=0.05)
    # bayesian_model = BayesianModel(model.edges)
    # print(bayesian_model.edges)
    edges = [('education.num', 'Income'), ('education.num', 'capital.loss'), ('capital.loss', 'Income'), ('capital.gain', 'Income'), ('capital.loss', 'age'), ('capital.loss', 'marital.status'), ('marital.status', 'age'),('Gender', 'Income'), ('race', 'native.country')]
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
        evidence_index = indeces.index(evidence_value)
        #evidence_value = int(evidence_values[evidence_key]-1)
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

    if attribute in interval_columns or q_k[0] in ordinal_columns:
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
def disc_score_one_instance(k, ordered_value_assignments, sens_attribute, dec_attribute, protected_label, unprotected_label):
    number_of_neighbours = 0
    index = 0
    protected_nearest_neighbours_labels = []
    unprotected_nearest_neighbours_lables = []
    while(number_of_neighbours < k):
        instances = ordered_value_assignments[index]
        protected_neighbours = instances[instances[sens_attribute] == protected_label]
        unprotected_neighbours = instances[instances[sens_attribute] == unprotected_label]
        neighbours_to_be_selected = min([len(protected_neighbours), len(unprotected_neighbours), k-number_of_neighbours])

        protected_nearest_neighbours_labels.extend(protected_neighbours[dec_attribute].iloc[0:neighbours_to_be_selected].values)
        unprotected_nearest_neighbours_lables.extend(unprotected_neighbours[dec_attribute].iloc[0:neighbours_to_be_selected].values)

        number_of_neighbours += neighbours_to_be_selected
        index += 1


    p1 = sum(unprotected_nearest_neighbours_lables)/k

    p2 = sum(protected_nearest_neighbours_labels)/k

    return p1-p2


def give_discrimination_scores_zhang(sens_attribute, decision_attribute, train_data, test_data, k, indices_info, range_dict, protected_label, unprotected_label):
    model = make_model_admission(train_data)
    model.fit(train_data, estimator=BayesianEstimator)
    cpd_decision = model.get_cpds(decision_attribute)
    cpd_sens = model.get_cpds(sens_attribute)
    Q, value_assignments = find_Q_value_assignments(model, sens_attribute, decision_attribute, train_data)
    instances_per_value_assignment = add_instances_to_value_assignments(value_assignments, train_data)

    protected_instances_test_set = test_data[sens_attribute] == protected_label
    protected_data_test = test_data[protected_instances_test_set]

    disc_scores = []
    for i in range(len(protected_data_test)):
        if protected_data_test[decision_attribute].iloc[i] == 1:
            disc_scores.append(0.0)
        else:
            distances = define_distances(Q, protected_data_test.iloc[i], value_assignments, instances_per_value_assignment, cpd_decision,
                                         cpd_sens, indices_info, range_dict, sens_attribute, decision_attribute)
            ordered_value_assignments = order_distances(instances_per_value_assignment, distances)
            disc_score = disc_score_one_instance(k, ordered_value_assignments, sens_attribute, decision_attribute, protected_label, unprotected_label)
            disc_scores.append(disc_score)

    return disc_scores


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
