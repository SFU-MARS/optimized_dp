import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from prediction.process_prediction_v2 import *

class ClusteringV2(object):

    def get_action(self):

        action_list_intersection, action_list_roundabout = ProcessPredictionV2().collect_action_from_group()

        acc_list = np.asarray([])
        omega_list = np.asarray([])

        # # Use intersection data
        # for action in action_list_intersection:
        #     acc_list = np.append(acc_list, action[1])
        #     omega_list = np.append(omega_list, action[2])

        # Use roundabout data
        for action in action_list_roundabout:
            acc_list = np.append(acc_list, action[1])
            omega_list = np.append(omega_list, action[2])

        return acc_list, omega_list

    def get_clustering(self):

        acc_list, omega_list = self.get_action()

        acc_omega = []
        num_in_effect = 0

        for i in range(len(acc_list)):
            # Filter those outside of physical bound
            if (-5 <= acc_list[i] <= 3) and (-0.34 <= omega_list[i] <= 0.34):
                acc_omega.append([acc_list[i], omega_list[i]])
                num_in_effect += 1

        print("the effective action number is %.2f" % (num_in_effect / len(acc_list)))
        print("total number of action is ", num_in_effect)

        acc_omega = np.asarray(acc_omega)

        pred = KMeans(n_clusters=9, random_state=0).fit_predict(acc_omega)
        plt.scatter(acc_omega[:, 0], acc_omega[:, 1], c=pred)
        plt.show()


if __name__ == "__main__":
    ClusteringV2().get_clustering()
