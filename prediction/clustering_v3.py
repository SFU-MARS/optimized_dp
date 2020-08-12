import sys
import pickle
sys.path.append("/Users/anjianli/Desktop/robotics/project/optimized_dp")

import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from prediction.process_prediction_v3 import *

class ClusteringV3(object):
    """
    Given cleaned data, we first normalize the acc and omega

    Then we set several default driving mode, and then cluster the variance

    """

    def __init__(self):

        self.clustering_num = 4

    def get_clustering(self):

        acc_list, omega_list = self.get_action()

        acc_omega, prediction = self.kmeans_clustering(acc_list, omega_list)

        self.plot_clustering(acc_omega, prediction)

    def get_action(self):

        action_list_intersection, action_list_roundabout = ProcessPredictionV3().collect_action_from_group()

        acc_list = np.asarray([])
        omega_list = np.asarray([])

        # Use intersection data
        for action in action_list_intersection:
            acc_list = np.append(acc_list, action[2])
            omega_list = np.append(omega_list, action[4] * 100)

        # # Use roundabout data
        # for action in action_list_roundabout:
        #     acc_list = np.append(acc_list, action[1])
        #     omega_list = np.append(omega_list, action[2])

        return acc_list, omega_list

    def kmeans_clustering(self, acc_list, omega_list):
        acc_omega = []
        num_in_effect = 0

        for i in range(len(acc_list)):
            acc_omega.append([acc_list[i], omega_list[i]])
            num_in_effect += 1

        print("the effective action number is %.2f" % (num_in_effect / len(acc_list)))
        print("total number of action is ", num_in_effect)

        acc_omega = np.asarray(acc_omega)

        # pred = KMeans(n_clusters=self.clustering_num, random_state=9).fit_predict(acc_omega)
        kmeans_action = KMeans(n_clusters=self.clustering_num, random_state=9).fit(acc_omega)
        pred = kmeans_action.predict(acc_omega)

        return acc_omega, pred

    def plot_clustering(self, data, prediction):

        fig, ax = plt.subplots()
        for i in range(self.clustering_num):
            ax.scatter(data[:, 0][prediction == i], data[:, 1][prediction == i], label='Cluster %d' % i)
        ax.set_xlabel('acceleration')
        ax.set_ylabel('angular_speed')
        ax.set_title('clustering_filter')
        # ax.set_title('clustering_no_filter')
        ax.legend()
        plt.show()

        # pickle.dump(kmeans_action, open("/home/anjianl/Desktop/project/optimized_dp/model/kmeans_action_intersection"
        #                                 ".pkl", "wb"))


if __name__ == "__main__":
    ClusteringV3().get_clustering()