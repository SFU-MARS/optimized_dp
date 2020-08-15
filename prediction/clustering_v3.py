import sys
import pickle
sys.path.append("/Users/anjianli/Desktop/robotics/project/optimized_dp")

import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from prediction.process_prediction_v3 import *

class ClusteringV3(object):
    """
    Given cleaned data, we first normalize the acc and omega

    Then we set several default driving mode, and then cluster the variance

    """

    def __init__(self):

        self.clustering_num = 5

        # Clustering feature selection
        # self.clustering_feature_type = "5_default"
        self.clustering_feature_type = "only_mean"
        # self.clustering_feature_type = "mean_and_variance"

        # Default driving mode
        # Decelerate
        self.default_m1_acc = -1.5
        self.default1_m1_omega = 0
        # Maintain
        self.default_m2_acc = 0
        self.default1_m2_omega = 0
        # Turn Left
        self.default_m3_acc = 0
        self.default1_m3_omega = 0.15
        # Turn right
        self.default_m4_acc = 0
        self.default1_m4_omega = - 0.15
        # Accelerate
        self.default_m5_acc = 1
        self.default1_m5_omega = 0

    def get_clustering(self):

        # The action feature vector is [acc_mean, acc_variance, omega_mean, omega_variance]
        action_feature_intersection, action_feature_list_roundabout = self.get_action_feature()

        # Form clustering feature from action feature
        clustering_feature_intersection = self.get_clustering_feature(action_feature_intersection)

        # Kmeans on clustering feature
        prediction = self.kmeans_clustering(clustering_feature_intersection)

        # Visualization
        self.plot_clustering(action_feature_intersection, clustering_feature_intersection, prediction)

    def get_action_feature(self):

        filename_action_feature_list_intersection, filename_action_feature_list_roundabout = ProcessPredictionV3().collect_action_from_group()

        # Concatenate all the actions feature in a big list
        action_feature_list_intersection = []
        action_feature_list_roundabout = []

        num_action_feature_intersection = 0
        for action_feature in filename_action_feature_list_intersection:
            for i in range(np.shape(action_feature[1])[0]):
                # The action feature vector is [acc_mean, acc_variance, omega_mean, omega_variance]
                action_feature_list_intersection.append([action_feature[1][i], action_feature[2][i], action_feature[3][i], action_feature[4][i]])
                num_action_feature_intersection += 1

        action_feature_list_intersection = np.asarray(action_feature_list_intersection)
        print("total number of action feature for intersection is ", num_action_feature_intersection)

        num_action_feature_roundabout = 0
        for action_feature in filename_action_feature_list_roundabout:
            for i in range(np.shape(action_feature[1])[0]):
                # The action feature vector is [acc_mean, acc_variance, omega_mean, omega_variance]
                action_feature_list_roundabout.append([action_feature[1][i], action_feature[2][i], action_feature[3][i], action_feature[4][i]])
                num_action_feature_roundabout += 1

        action_feature_list_roundabout = np.asarray(action_feature_list_roundabout)
        print("total number of action feature for roundabout is ", num_action_feature_roundabout)

        return action_feature_list_intersection, action_feature_list_roundabout

    def get_clustering_feature(self, action_feature):

        # action_feature = [acc_mean, acc_variance, omega_mean, omega_variance]
        if self.clustering_feature_type == "5_default":
            clustering_feature = np.transpose(np.asarray([
                action_feature[:, 0] - self.default_m1_acc, action_feature[:, 0] - self.default_m2_acc,
                action_feature[:, 0] - self.default_m3_acc, action_feature[:, 0] - self.default_m4_acc,
                action_feature[:, 0] - self.default_m5_acc,
                action_feature[:, 2] - self.default1_m1_omega, action_feature[:, 2] - self.default1_m2_omega,
                action_feature[:, 2] - self.default1_m3_omega, action_feature[:, 2] - self.default1_m4_omega,
                action_feature[:, 2] - self.default1_m5_omega]))
        elif self.clustering_feature_type == "only_mean":
            clustering_feature = np.transpose(np.asarray([action_feature[:, 0], action_feature[:, 2]]))
        elif self.clustering_feature_type == "mean_and_variance":
            clustering_feature = np.transpose(np.asarray([action_feature[:, 0], action_feature[:, 2],
                                                          action_feature[:, 1], action_feature[:, 3]]))

        normalized_clustering_feature = MinMaxScaler().fit_transform(clustering_feature)

        return normalized_clustering_feature

    def kmeans_clustering(self, clustering_feature):

        kmeans_action = KMeans(n_clusters=self.clustering_num, random_state=9).fit(clustering_feature)
        pred = kmeans_action.predict(clustering_feature)

        return pred

    def plot_clustering(self, original_data, clustering_data, prediction):

        fig, ax = plt.subplots()
        for i in range(self.clustering_num):
            ax.scatter(original_data[:, 0][prediction == i], original_data[:, 2][prediction == i], label='Cluster %d' % i)
        ax.set_xlabel('acceleration')
        ax.set_ylabel('angular_speed')
        ax.set_title(self.clustering_feature_type)
        ax.legend()
        plt.show()

        # pickle.dump(kmeans_action, open("/home/anjianl/Desktop/project/optimized_dp/model/kmeans_action_intersection"
        #                                 ".pkl", "wb"))


if __name__ == "__main__":
    ClusteringV3().get_clustering()