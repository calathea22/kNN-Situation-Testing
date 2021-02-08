import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import math
from scipy.linalg import sqrtm
from load_data import load_data, load_optimization_info
import pandas as pd
import utils

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_protected_vs_unprotected(data, protected_info, class_label, title):
    protected_data = data.iloc[np.where(protected_info == 1)[0]]
    unprotected_data = data.iloc[np.where(protected_info == 2)[0]]

    protected_labels = class_label[np.where(protected_info == 1)[0]]
    unprotected_labels = class_label[np.where(protected_info == 2)[0]]

    cmap_protected = matplotlib.colors.ListedColormap(['salmon', 'maroon'])
    cmap_unprotected = matplotlib.colors.ListedColormap(['cornflowerblue', 'navy'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(protected_data[0], protected_data[1], protected_data[2], c=protected_labels, cmap=cmap_protected, marker='o', label='Women', alpha=0.2)
    ax.scatter(unprotected_data[0], unprotected_data[1], unprotected_data[2], c=unprotected_labels, cmap=cmap_unprotected, marker='o', label='Men', alpha=0.2)


    # ax.scatter(unprotected_data[0], unprotected_data[1], unprotected_data[2], c=unprotected_labels, cmap=cmap_unprotected, marker='o', label='Men', alpha=0.2)

    set_axes_equal(ax)
    ax.set_xlabel('Height')
    ax.set_ylabel('Score')
    ax.set_zlabel('Extra curricular')
    ax.legend()
    ax.set_title(title)

    plt.show()
    return

def visualize_positive_vs_negative(data, protected_info, class_label, title):
    positive_class_data = data.iloc[np.where(class_label == 1)[0]]
    negative_class_data = data.iloc[np.where(class_label == 0)[0]]
    # negative_class_and_discriminated = data.iloc[discriminated_instances]

    protected_positive_class_info = protected_info[np.where(class_label == 1)[0]]
    protected_negative_class_info = protected_info[np.where(class_label == 0)[0]]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cmap_decision = matplotlib.colors.ListedColormap(['indianred', 'navy'])

    ax.scatter(positive_class_data.iloc[:,0], positive_class_data.iloc[:,1], positive_class_data.iloc[:,2], c=protected_positive_class_info, cmap=cmap_decision,
               marker='+',alpha=0.2)
    ax.scatter(negative_class_data.iloc[:,0], negative_class_data.iloc[:,1], negative_class_data.iloc[:,2], c=protected_negative_class_info,
               cmap=cmap_decision, marker='_', alpha=0.2)
    # ax.scatter(negative_class_and_discriminated.iloc[:,0], negative_class_and_discriminated.iloc[:, 1], negative_class_and_discriminated.iloc[:,2],
    #            c='black', marker='_', alpha=1)

    markers = ["+", "_"]
    colors = ['indianred', 'blue']

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    handles = [f("s", colors[i]) for i in range(2)]
    handles += [f(markers[i], "k") for i in range(2)]


    set_axes_equal(ax)
    ax.set_xlabel('Height')
    ax.set_ylabel('Score')
    ax.set_zlabel('Extra curricular')
    ax.legend(handles, ["Women", "Men", "Admitted", "Not Admitted"])
    ax.set_title(title)

    plt.show()
    return

def visualize(data, protected_info, class_label, title):
    positive_class_labels_data = data.iloc[np.where(class_label == 1)[0]]
    negative_class_labels_data = data.iloc[np.where(class_label == 2)[0]]
    protected_data = data.iloc[np.where(protected_info == 1)[0]]
    unprotected_data = data.iloc[np.where(protected_info == 2)[0]]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    ax1.scatter(positive_class_labels_data[0], positive_class_labels_data[1], positive_class_labels_data[2],
               c='limegreen',
               marker='o', label="Positive Class", alpha=0.1)
    ax1.scatter(negative_class_labels_data[0], negative_class_labels_data[1], negative_class_labels_data[2],
               c="darkorange", label="Negative Class", marker='o', alpha=0.1)

    set_axes_equal(ax1)
    ax1.set_xlabel('Height')
    ax1.set_ylabel('Score')
    ax1.set_zlabel('Extra curricular')
    ax1.legend()
    ax1.set_title(title)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax2.scatter(protected_data[0], protected_data[1], protected_data[2], c='indianred', marker='o', label='Women', alpha=0.2)
    ax2.scatter(unprotected_data[0], unprotected_data[1], unprotected_data[2], c='slateblue', marker='o', label='Men', alpha=0.2)

    #cmap_unprotected = matplotlib.colors.ListedColormap(['slateblue'])
    # ax.scatter(unprotected_data[0], unprotected_data[1], unprotected_data[2], c=unprotected_labels, cmap=cmap_unprotected, marker='o', label='Men', alpha=0.2)
    set_axes_equal(ax2)
    ax2.set_xlabel('Height')
    ax2.set_ylabel('Score')
    ax2.set_zlabel('Extra curricular')
    ax2.legend()
    ax2.set_title(title)

    plt.show()
    return

def visualize_baseline(data_location, title):
    data_dict = load_data(data_location, "train")

    data = data_dict['data']
    print(data)
    protected_info = data_dict['protected_info']
    class_label = data_dict['class_label']
    discriminated_instances = data_dict['discriminated_instances']

    visualize_positive_vs_negative(data, protected_info, class_label, discriminated_instances, title)

def visualize_luong(data_location, title):
    data_dict = load_data(data_location, "train")

    standardized_data = data_dict['standardized_data']
    protected_info = data_dict['protected_info']
    class_label = data_dict['class_label']
    discriminated_instances = data_dict['discriminated_instances']


    visualize_positive_vs_negative(standardized_data, protected_info, class_label, discriminated_instances, title)


def visualize_euclidean(data_location, title):
    data_dict = load_data(data_location, "train")
    loaded_optimization_info = load_optimization_info(data_location, lambda_l1_norm_euclidean=0.2, lambda_l1_norm_mahalanobis=0.2)
    standardized_data = data_dict['standardized_data']
    protected_info = data_dict['protected_info']
    class_label = data_dict['class_label']
    discriminated_instances = data_dict['discriminated_instances']

    weights_euclidean = loaded_optimization_info['weights_euclidean']

    projected_data = utils.project_to_weighted_euclidean(standardized_data, weights_euclidean)

    visualize_positive_vs_negative(projected_data, protected_info, class_label, discriminated_instances, title)


def mahalanobis_distance(x, y, weight_array):
    weight_matrix = np.reshape(weight_array, (len(x), len(x)))
    abs_difference = abs(x-y)
    print(abs_difference)
    transposed_difference = np.transpose(abs_difference)
    print(transposed_difference)
    dot_product1 = np.matmul(transposed_difference, weight_matrix)
    print(dot_product1)
    return math.sqrt(np.matmul(dot_product1, abs_difference))


def euclidean_distance(x, y):
    sum_of_distances = 0
    for i in range(0, len(x)):
        sum_of_distances += (x[i]-y[i])**2
    return math.sqrt(sum_of_distances)


def visualize_mahalanobis(data_location, lambda_l1_norm, title):
    data_dict = load_data(data_location, "train")
    loaded_optimization_info = load_optimization_info(data_location, lambda_l1_norm_euclidean=0.2, lambda_l1_norm_mahalanobis=lambda_l1_norm)
    mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']
    standardized_data = data_dict['standardized_data']
    protected_info = data_dict['protected_info']
    class_label = data_dict['class_label']
    discriminated_instances = data_dict['discriminated_instances']
    # mahalanobis_array = loaded_optimization_info['mahalanobis_matrix']


    projected_data = utils.project_to_mahalanobis(standardized_data, mahalanobis_matrix)
    visualize_positive_vs_negative(projected_data, protected_info, class_label, title)



