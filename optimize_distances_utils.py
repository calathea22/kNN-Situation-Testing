import numpy as np


def metric_to_linear(M):
    """
    Converts a metric PSD matrix into an associated linear transformation matrix, so the distance defined by the
    metric matrix is the same as the euclidean distance after projecting by the linear transformation.
    This implementation takes the linear transformation corresponding to the square root of the matrix M.
    Parameters
    ----------
    M : 2D-Array or Matrix
        A positive semidefinite matrix.
    Returns
    -------
    L : 2D-Array
        The matrix associated to the linear transformation that computes the same distance as M.
    """
    eigvals, eigvecs = np.linalg.eig(M)
    eigvals = eigvals.astype(float)  # Remove residual imaginary part
    eigvecs = eigvecs.astype(float)
    eigvals[eigvals < 0.0] = 0.0  # MEJORAR ESTO (no debería hacer falta, pero está bien para errores de precisión)
    sqrt_diag = np.sqrt(eigvals)
    return eigvecs.dot(np.diag(sqrt_diag)).T


def SDProject(M):
    """
    Projects a symmetric matrix onto the positive semidefinite cone.
    The projection is made by taking the non negative eigenvalues after diagonalizing.
    Parameters
    ----------
    M : 2D-Array or Matrix
        A symmetric matrix.
    Returns
    -------
    Mplus : 2D-Array
        The projection of M onto the positive semidefinite cone.
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = eigvals.astype(float)
    eigvecs = eigvecs.astype(float)
    eigvals[eigvals < 0.0] = 0.0
    diag_sdp = np.diag(eigvals)
    return eigvecs.dot(diag_sdp).dot(eigvecs.T)


def calc_distances_within_and_between_classes(labels, data, weights, indices_info, distance_function):
    dist_diff_classes = []
    dist_same_classes = []
    for i in range(0, len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                dist_diff_classes.append(distance_function(data[i], data[j], weights, indices_info))
            else:
                dist_same_classes.append(distance_function(data[i], data[j], weights, indices_info))
    return (dist_diff_classes, dist_same_classes)


def get_abs_difference_between_instances_with_same_and_different_class_label(labels, data, indices_info):
    same_classes = []
    different_classes = []

    for i in range(0, len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                different_classes.append(give_abs_difference_vector_between_instances(data[i], data[j], indices_info))
            else:
                same_classes.append(give_abs_difference_vector_between_instances(data[i], data[j], indices_info))
    return(same_classes, different_classes)


def get_non_abs_difference_between_instances_with_same_and_different_class_label(labels, data, indices_info):
    same_classes = []
    different_classes = []

    for i in range(0, len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                different_classes.append(give_non_abs_difference_vector_between_instances(data[i], data[j], indices_info))
            else:
                same_classes.append(give_non_abs_difference_vector_between_instances(data[i], data[j], indices_info))
    return(same_classes, different_classes)


def give_abs_difference_vector_between_instances(x, y, indices_info):
    interval_indices = indices_info['interval']
    ordinal_indices = indices_info['ordinal']

    difference_vector = []
    for index in range(0, len(x)):
        if index in interval_indices or index in ordinal_indices:
            difference_vector.append(abs(x[index]-y[index]))
        else:
            difference_vector.append(x[index] != y[index])
    return np.array(difference_vector)


def give_non_abs_difference_vector_between_instances(x, y, indices_info):
    interval_indices = indices_info['interval']
    ordinal_indices = indices_info['ordinal']

    difference_vector = []
    for index in range(0, len(x)):
        if index in interval_indices or index in ordinal_indices:
            difference_vector.append(x[index]-y[index])
        else:
            difference_vector.append(x[index] != y[index])
    return np.array(difference_vector)
