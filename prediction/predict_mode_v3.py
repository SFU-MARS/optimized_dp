import numpy as np
import random
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/anjianli/Desktop/robotics/project/optimized_dp")

from prediction.clustering_v3 import ClusteringV3
from prediction.process_prediction_v3 import ProcessPredictionV3

class PredictModeV3(object):

    def __init__(self):

        # Which scenario to predict
        self.scenario_predict = "intersection"

        # Data directory
        # Remote desktop
        # self.file_dir_intersection = '/home/anjianl/Desktop/project/optimized_dp/data/intersection-data'
        # self.file_dir_roundabout = '/home/anjianl/Desktop/project/optimized_dp/data/roundabout-data'
        # My laptop
        self.file_dir_intersection = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/intersection-data'
        self.file_dir_roundabout = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/roundabout-data'

        # File name
        self.file_name_intersection = ['car_16_vid_09.csv', 'car_20_vid_09.csv', 'car_29_vid_09.csv',
                                       'car_36_vid_11.csv', 'car_50_vid_03.csv', 'car_112_vid_11.csv',
                                       'car_122_vid_11.csv',
                                       'car_38_vid_02.csv', 'car_52_vid_07.csv', 'car_73_vid_02.csv',
                                       'car_118_vid_11.csv']
        self.file_name_roundabout = ['car_27.csv', 'car_122.csv',
                                     'car_51.csv', 'car_52.csv', 'car_131.csv', 'car_155.csv']

    def predict_mode(self):

        # Format of action_bound_mode is
        # [Mode name, acc_min, acc_max, omega_min, omega_max]
        self.action_bound_mode = ClusteringV3().get_clustering()

        for file in self.file_name_intersection:
            # Get raw action data from traj file
            raw_acc_list, raw_omega_list = self.get_predict_traj(scenario=self.scenario_predict, traj_file_pred=file)

            # How to deal with outliers
            filter_acc_list, filter_omega_list = self.filter_action(raw_acc_list, raw_omega_list)

            # Get mode and plot
            for i in range(len(filter_acc_list)):
                filter_acc, filter_omega = filter_acc_list[i], filter_omega_list[i]

                mode_num_seq, mode_num_str = self.get_mode(filter_acc, filter_omega)

                self.plot_mode(mode_num_seq, mode_num_str)

    def get_predict_traj(self, scenario, traj_file_pred=None):

        if traj_file_pred is None:
            if scenario == "intersection":
                # If not specified, then randomly pick a file
                random.seed(13)
                index = random.randint(0, len(self.file_name_intersection) - 1)
                traj_file_name = self.file_dir_intersection + '/' + self.file_name_intersection[index]
                print("the traj file to predict is", traj_file_name)

            traj_file = ProcessPredictionV3().read_prediction(file_name=traj_file_name)
        else:
            if scenario == "intersection":
                traj_file_name = self.file_dir_intersection + '/' + traj_file_pred
                print("the traj file to predict is", traj_file_name)

            traj_file = ProcessPredictionV3().read_prediction(file_name=traj_file_name)

        raw_traj = ProcessPredictionV3().extract_traj(traj_file)

        # Fit polynomial for x, y position: x(t), y(t)
        poly_traj = ProcessPredictionV3().fit_polynomial_traj(raw_traj)

        if ProcessPredictionV3().use_velocity:
            # Get the acc from velocity profile provided
            raw_acc_list, raw_omega_list = ProcessPredictionV3().get_action_v_profile(raw_traj, poly_traj)
        else:
            # Get raw actions from poly_traj, here acc and omega are extracted from both poly_traj
            raw_acc_list, raw_omega_list = ProcessPredictionV3().get_action_poly(poly_traj)

        return raw_acc_list, raw_omega_list

    def filter_action(self, raw_acc_list, raw_omega_list):

        filter_acc_list = []
        filter_omega_list = []

        for i in range(len(raw_acc_list)):
            if np.shape(raw_acc_list[i])[0] < ProcessPredictionV3().mode_time_span:
                print("not qualified", np.shape(raw_acc_list[i])[0])
                continue
            # print("raw omega", raw_omega_list[i])
            acc_interpolate, omega_interpolate = ProcessPredictionV3().to_interpolate(raw_acc_list[i], raw_omega_list[i])
            print("acc size", np.shape(acc_interpolate)[0])
            # print("filter omega", omega_interpolate)
            filter_acc_list.append(acc_interpolate)
            filter_omega_list.append(omega_interpolate)

        return filter_acc_list, filter_omega_list

    def get_mode(self, raw_acc, raw_omega):

        mode_num_seq = []
        mode_num_str = []
        for i in range(np.shape(raw_acc)[0]):
            if i + ProcessPredictionV3().mode_time_span <= np.shape(raw_acc)[0]:
                curr_mode_num, curr_mode_str = self.decide_mode(raw_acc[i:i + ProcessPredictionV3().mode_time_span],
                                                                raw_omega[i:i + ProcessPredictionV3().mode_time_span])
                # print(curr_mode_str)
                mode_num_seq.append(curr_mode_num)
                mode_num_str.append(curr_mode_str)
            else:
                mode_num_seq.append(curr_mode_num)
                mode_num_str.append(curr_mode_str)

        return np.asarray(mode_num_seq), mode_num_str

    def decide_mode(self, acc, omega):

        if (np.shape(acc)[0] != ProcessPredictionV3().mode_time_span) or (np.shape(omega)[0] != ProcessPredictionV3().mode_time_span):
            print("prediction dimension is wrong")
            return 0

        acc_mean = np.mean(acc)
        omega_mean = np.mean(omega)

        for i in range(len(self.action_bound_mode)):
            if (self.action_bound_mode[i][1] <= acc_mean <= self.action_bound_mode[i][2]) and (self.action_bound_mode[i][3] <= omega_mean <= self.action_bound_mode[i][4]):
                return i, self.action_bound_mode[i][0]

        return -1, "mode -1"

    def plot_mode(self, mode_num_seq, mode_num_str):

        fig, ax = plt.subplots()

        time_index = np.linspace(0, np.shape(mode_num_seq)[0], num=np.shape(mode_num_seq)[0])
        # print(mode_num_seq)
        # print(time_index)

        ax.plot(time_index, mode_num_seq, 'o-')
        ax.grid()

        locs, labels = plt.xticks()
        plt.xticks(np.arange(0, np.shape(mode_num_seq)[0], step=10))
        plt.show()


if __name__ == "__main__":
    PredictModeV3().predict_mode()
