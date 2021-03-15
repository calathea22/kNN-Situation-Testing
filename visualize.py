import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import math
from scipy.linalg import sqrtm
from load_data import load_data, load_optimization_info
import pandas as pd
import utils
from kNN_discrimination_discovery import weighted_euclidean_distance, mahalanobis_distance, luong_distance
from seaborn import boxplot
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
# plt.rcParams['font.family'] = 'Avenir'

# plt.rcParams['axes.linewidth'] = 2
plt.style.use(['science', 'no-latex'])
#plt.rcParams['font.size'] = 10


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
    ax.set_ylabel('Skills')
    ax.set_zlabel('Workinghours')
    ax.legend(handles, ["Women", "Men", "High Wage", "Low Wage"])
    ax.set_title(title)

    plt.show()
    return


def visualize_positive_vs_negative_with_rejected(train_data, train_protected_info, val_data, val_protected_info, rejected_indices, title):
    unprotected_class_data_train = train_data.iloc[np.where(train_protected_info == 2)[0]]

    rejected_data_val = val_data.iloc[rejected_indices]
    val_protected_indices = np.where(val_protected_info == 1)[0]
    val_protected_not_rejected_indices = set(val_protected_indices) - set(rejected_indices)
    print(val_protected_not_rejected_indices)
    rest_of_protected_data_val = val_data.iloc[list(val_protected_not_rejected_indices)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(unprotected_class_data_train.iloc[:,0], unprotected_class_data_train.iloc[:,1], unprotected_class_data_train.iloc[:,2], c='navy', alpha=0.1)
    ax.scatter(rest_of_protected_data_val.iloc[:, 0], rest_of_protected_data_val.iloc[:, 1], rest_of_protected_data_val.iloc[:, 2], c='indianred', alpha=0.1)
    ax.scatter(rejected_data_val.iloc[:,0], rejected_data_val.iloc[:,1], rejected_data_val.iloc[:,2], c='black', alpha=0.2)

    #hier iets toevoegen dat alle rejected instances ander kleurtje krijgen of iets dergelijks
    #set_axes_equal(ax)
    ax.set_xlabel('Height')
    ax.set_ylabel('Skills')
    ax.set_zlabel('Workinghours')
    ax.set_title(title)

    plt.show()
    return

def rand_jitter(arr):
    stdev = .013 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def visualize_positive_vs_negative_with_rejected_2d(train_data, train_protected_info, train_class_label, val_data, val_protected_info, val_class_label, rejected_indices, title):
    unprotected_data_train = train_data.iloc[np.where(train_protected_info == 2)[0]]
    train_class_label_unprotected = train_class_label[np.where(train_protected_info == 2)[0]]
    unprotected_high_wage_data = unprotected_data_train.iloc[np.where(train_class_label_unprotected == 1)[0]]
    unprotected_low_wage_data = unprotected_data_train.iloc[np.where(train_class_label_unprotected == 0)[0]]

    rejected_data_val = val_data.iloc[rejected_indices]

    val_protected_indices = np.where(val_protected_info == 1)[0]
    negative_class_labels_protected_indices = np.where(val_class_label == 0)[0]
    positive_class_labels_protected_indices = np.where(val_class_label==1)[0]
    protected_indices_with_negative_class_label = set(val_protected_indices) & set(negative_class_labels_protected_indices)
    protected_indices_with_positive_class_label = set(val_protected_indices) & set(positive_class_labels_protected_indices)

    val_protected_not_rejected_indices_with_neg_label = protected_indices_with_negative_class_label - set(rejected_indices)
    print(val_protected_not_rejected_indices_with_neg_label)
    not_rejected_protected_with_neg_decision = val_data.iloc[list(val_protected_not_rejected_indices_with_neg_label)]
    not_rejected_protected_with_pos_decision = val_data.iloc[list(protected_indices_with_positive_class_label)]

    plt.scatter(rand_jitter(unprotected_low_wage_data.iloc[:, 2]), rand_jitter(unprotected_low_wage_data.iloc[:, 1]), c='navy', alpha=0.6,
                marker="_")
    plt.scatter(rand_jitter(unprotected_high_wage_data.iloc[:, 2]), rand_jitter(unprotected_high_wage_data.iloc[:, 1]), c='navy', alpha=0.6,
                marker="+")

    plt.scatter(rand_jitter(not_rejected_protected_with_neg_decision.iloc[:, 2]),
                rand_jitter(not_rejected_protected_with_neg_decision.iloc[:, 1]), c='indianred', alpha=0.6, marker="_")
    plt.scatter(rand_jitter(not_rejected_protected_with_pos_decision.iloc[:, 2]),
                rand_jitter(not_rejected_protected_with_pos_decision.iloc[:, 1]), c='indianred', alpha=0.6, marker="+")

    plt.scatter(rand_jitter(rejected_data_val.iloc[:,2]), rand_jitter(rejected_data_val.iloc[:,1]),  c='grey', alpha=0.8, marker="_")

    #hier iets toevoegen dat alle rejected instances ander kleurtje krijgen of iets dergelijks
    # plt.axis('equal')
    ax = plt.gca()
    colors = ['navy', 'indianred', 'grey']

    circles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8) for c in colors]
    legend1 = plt.legend(circles, ['Men', 'Women not rejected', 'Women rejected'], loc='upper right', frameon=False)
    ax.add_artist(legend1)

    decision_markes = [Line2D([0], [0], marker='+', color='w', markerfacecolor='black', markeredgecolor ='black', markersize=8), Line2D([0], [0], marker='_', color='w', markerfacecolor='black', markeredgecolor ='black', markersize=8)]
    legend2 = plt.legend(decision_markes, ['High Wage', 'Low Wage'], loc='upper left', frameon=False)
    ax.add_artist(legend2)

    plt.xlabel('Workinghours')
    plt.ylabel('Skills')
    plt.title(title)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)


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


    visualize_positive_vs_negative(standardized_data, protected_info, class_label, title)


def visualize_euclidean(data_location, lambda_l1_norm, title):
    data_dict = load_data(data_location, "train")
    loaded_optimization_info = load_optimization_info(data_location, lambda_l1_norm_euclidean=lambda_l1_norm, lambda_l1_norm_mahalanobis=lambda_l1_norm)
    standardized_data = data_dict['standardized_data']
    protected_info = data_dict['protected_info']
    class_label = data_dict['class_label']
    discriminated_instances = data_dict['discriminated_instances']

    weights_euclidean = loaded_optimization_info['weights_euclidean']

    projected_data = utils.project_to_weighted_euclidean(standardized_data, weights_euclidean)

    visualize_positive_vs_negative(projected_data, protected_info, class_label, title)

    return

# def mahalanobis_distance(x, y, weight_array):
#     weight_matrix = np.reshape(weight_array, (len(x), len(x)))
#     abs_difference = abs(x-y)
#     print(abs_difference)
#     transposed_difference = np.transpose(abs_difference)
#     print(transposed_difference)
#     dot_product1 = np.matmul(transposed_difference, weight_matrix)
#     print(dot_product1)
#     return math.sqrt(np.matmul(dot_product1, abs_difference))

#
# def euclidean_distance(x, y):
#     sum_of_distances = 0
#     for i in range(0, len(x)):
#         sum_of_distances += (x[i]-y[i])**2
#     return math.sqrt(sum_of_distances)


def visualize_mahalanobis(data_location, lambda_l1_norm, title):
    data_dict = load_data(data_location, "train")
    loaded_optimization_info = load_optimization_info(data_location, lambda_l1_norm_euclidean=lambda_l1_norm, lambda_l1_norm_mahalanobis=lambda_l1_norm)
    mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']
    standardized_data = data_dict['standardized_data']
    protected_info = data_dict['protected_info']
    class_label = data_dict['class_label']
    discriminated_instances = data_dict['discriminated_instances']
    # mahalanobis_array = loaded_optimization_info['mahalanobis_matrix']

    print(mahalanobis_matrix)
    projected_data = utils.project_to_mahalanobis(standardized_data, mahalanobis_matrix)
    print(projected_data)
    #visualize_positive_vs_negative(projected_data, protected_info, class_label, title)


def visualize_rejected_instances(data_location, rejected_indices, title):
    data_dict_train = load_data(data_location, "train")
    data_dict_val = load_data(data_location, "val")

    standardized_data_train = data_dict_train['standardized_data']
    protected_info_train = data_dict_train['protected_info']
    class_label_train = data_dict_train['class_label']

    standardized_data_val = data_dict_val['standardized_data']
    protected_info_val = data_dict_val['protected_info']
    class_label_val = data_dict_val['class_label']

    visualize_positive_vs_negative_with_rejected_2d(standardized_data_train, protected_info_train, class_label_train, standardized_data_val, protected_info_val, class_label_val, rejected_indices, title)


def visualize_inter_and_intra_distances(location, lambda_l1_euclidean, lambda_l1_mahalanobis, indices_info):
    loaded_data = load_data(location, "train")
    loaded_optimization_info = load_optimization_info(location, lambda_l1_euclidean, lambda_l1_mahalanobis)

    standardized_data = loaded_data['standardized_data']
    protected_info = loaded_data['protected_info']

    euclidean_weights = loaded_optimization_info['weights_euclidean']
    mahalanobis_matrix = loaded_optimization_info['mahalanobis_matrix']

    luong_distances = utils.make_distance_matrix_based_on_distance_function(standardized_data, luong_distance, [], indices_info)
    weighted_euclidean_distances = utils.make_distance_matrix_based_on_distance_function(standardized_data, weighted_euclidean_distance, euclidean_weights, indices_info)
    mahalanobis_distances = utils.make_distance_matrix_based_on_distance_function(standardized_data, mahalanobis_distance, mahalanobis_matrix, indices_info)

    # print("BASELINE")
    # inter_prot_base, inter_unprot_base, intra_base = utils.get_inter_and_intra_sens_distances(baseline_distances,
    #                                                                                           protected_info,
    #                                                                                           protected_label)
    print("Luong")
    inter_prot_luong, inter_unprot_luong, intra_luong = utils.get_inter_and_intra_sens_distances(luong_distances,
                                                                                                 protected_info,
                                                                                                 1)
    # print("Zhang")
    # inter_prot_zhang, inter_unprot_zhang, intra_zhang = utils.get_inter_and_intra_sens_distances(zhang_distances,
    #                                                                                              protected_info,
    #                                                                                              protected_label)
    print("Weighted Euclidean")
    inter_prot_euclidean, inter_unprot_euclidean, intra_euclidean = utils.get_inter_and_intra_sens_distances(
        weighted_euclidean_distances, protected_info, 1)
    print("Mahalanobis")
    inter_prot_mahalanobis, inter_unprot_mahalanobis, intra_mahalanobis = utils.get_inter_and_intra_sens_distances(
        mahalanobis_distances, protected_info, 1)

    distances_inter_prot = pd.DataFrame(columns=["Luong", "Euclidean", "Mahalanobis"])

    # distances_inter_prot['Baseline'] = inter_prot_base
    distances_inter_prot['Luong'] = inter_prot_luong
    # distances_inter_prot['Zhang'] = inter_prot_zhang
    distances_inter_prot['Euclidean'] = inter_prot_euclidean
    distances_inter_prot['Mahalanobis'] = inter_prot_mahalanobis
    distances_inter_prot_melted = pd.melt(distances_inter_prot)
    distances_inter_prot_melted['Cluster'] = "Women vs. Women"

    distances_inter_unprot = pd.DataFrame(columns=["Luong", "Euclidean", "Mahalanobis"])
    # distances_inter_unprot['Baseline'] = inter_unprot_base
    distances_inter_unprot['Luong'] = inter_unprot_luong
    # distances_inter_unprot['Zhang'] = inter_unprot_zhang
    distances_inter_unprot['Euclidean'] = inter_unprot_euclidean
    distances_inter_unprot['Mahalanobis'] = inter_unprot_mahalanobis
    distances_inter_unprot_melted = pd.melt(distances_inter_unprot)
    distances_inter_unprot_melted['Cluster'] = "Men vs. Men"

    distances_intra = pd.DataFrame(columns=["Luong", "Euclidean", "Mahalanobis"])
    # distances_intra['Baseline'] = intra_base
    distances_intra['Luong'] = intra_luong
    # distances_intra['Zhang'] = intra_zhang
    distances_intra['Euclidean'] = intra_euclidean
    distances_intra['Mahalanobis'] = intra_mahalanobis
    distances_intra_melted = pd.melt(distances_intra)
    distances_intra_melted['Cluster'] = "Men vs. Women"

    all_distances = distances_inter_prot_melted.append(distances_inter_unprot_melted)
    all_distances = all_distances.append(distances_intra_melted)
    all_distances = all_distances.rename(columns={'variable': 'Measure', 'value': 'Distance'}, inplace=False)

    boxplot(x="Cluster", y="Distance", hue='Measure', data=all_distances, showmeans=True,
            meanprops={"marker": "o",
                       "markerfacecolor": "white",
                       "markeredgecolor": "black",
                       "markersize": "10"})
    plt.xlabel("Gender clusters", size=14)
    plt.ylabel("Distance", size=14)
    plt.title("Inter- and intra distances within and between genders", size=18)
    plt.legend(loc='upper right')
    plt.show()