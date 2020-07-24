import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from prediction.process_prediction import *

class Clustering(object):

    def get_action_bound(self):

        action_bound_list_intersection, action_bound_list_roundabout = ProcessPrediction().collect_action_bound_from_group()

        acc_bound = []
        ang_v_bound = []

        # Use intersection data
        for bound in action_bound_list_intersection:
            acc_bound.append([bound[1], bound[2]])
            ang_v_bound.append([bound[3], bound[4]])

        # # Use roundabout data
        # for bound in action_bound_list_roundabout:
        #     acc_bound.append([bound[1], bound[2]])
        #     ang_v_bound.append([bound[3], bound[4]])

        return np.asarray(acc_bound), np.asarray(ang_v_bound)

    def get_clustering(self):

        acc_bound, ang_v_bound = self.get_action_bound()

        # plt.scatter(acc_bound[:, 0], acc_bound[:, 1], marker='o')
        # plt.show()

        acc_pred = KMeans(n_clusters=4, random_state=0).fit_predict(acc_bound)

        # plt.scatter(acc_bound[:, 0], acc_bound[:, 1], c=acc_pred)
        # plt.show()

        ang_v_pred = KMeans(n_clusters=4, random_state=9).fit_predict(ang_v_bound)
        plt.scatter(ang_v_bound[:, 0], ang_v_bound[:, 1], c=ang_v_pred)
        plt.show()


if __name__ == "__main__":
    Clustering().get_clustering()
