import numpy as np
import os
from os import path
import random
import matplotlib.pyplot as plt

import math

import sys
sys.path.append("/Users/anjianli/Desktop/robotics/project/optimized_dp")

from prediction.clustering_v3 import ClusteringV3
from prediction.process_prediction_v3 import ProcessPredictionV3

class PredictModeV3(object):

    def __init__(self):

        # self.to_save_pred_mode = True
        self.to_plot_pred_mode = True

        self.to_save_pred_mode = False
        # self.to_plot_pred_mode = False

        # Which scenario to predict
        # self.scenario_predict = "intersection"
        self.scenario_predict = "roundabout"

        # Data directory
        # Remote desktop
        # if self.scenario_predict == "intersection":
        #     self.file_dir_predict = '/home/anjianl/Desktop/project/optimized_dp/data/intersection-data'
        # elif self.scenario_predict == "roundabout":
        #     self.file_dir_predict = '/home/anjianl/Desktop/project/optimized_dp/data/roundabout-data'
        # My laptop
        if self.scenario_predict == "intersection":
            self.file_dir_predict = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/intersection-data'
        elif self.scenario_predict == "roundabout":
            self.file_dir_predict = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/roundabout-data'

        # File name
        if self.scenario_predict == "intersection":
            self.file_name_predict = ['car_16_vid_09.csv', 'car_20_vid_09.csv', 'car_29_vid_09.csv',
                                           'car_36_vid_11.csv', 'car_50_vid_03.csv', 'car_112_vid_11.csv',
                                           'car_122_vid_11.csv',
                                           'car_38_vid_02.csv', 'car_52_vid_07.csv', 'car_73_vid_02.csv',
                                           'car_118_vid_11.csv']
        elif self.scenario_predict == "roundabout":
            self.file_name_predict = ['car_27.csv', 'car_122.csv',
                                      'car_51.csv', 'car_52.csv', 'car_131.csv', 'car_155.csv',
                                      'car_15.csv', 'car_28.csv', 'car_34.csv', 'car_41.csv', 'car_50.csv',
                                      'car_61.csv', 'car_75.csv', 'car_80.csv']

    def predict_mode(self):

        # Format of action_bound_mode is
        # [Mode name, acc_min, acc_max, omega_min, omega_max]
        self.action_bound_mode = ClusteringV3().get_clustering()

        print(self.action_bound_mode)

        for file in self.file_name_predict:
            # Get raw action data from traj file
            raw_acc_list, raw_omega_list = self.get_predict_traj(scenario=self.scenario_predict, traj_file_pred=file)

            # How to deal with outliers
            filter_acc_list, filter_omega_list = self.filter_action(raw_acc_list, raw_omega_list)

            # Get mode and plot
            for i in range(len(filter_acc_list)):
                filter_acc, filter_omega = filter_acc_list[i], filter_omega_list[i]

                mode_num_seq, mode_num_str = self.get_mode(filter_acc, filter_omega)

                if self.to_plot_pred_mode:
                    self.plot_mode(mode_num_seq, mode_num_str, filter_acc, filter_omega)

                if self.to_save_pred_mode:
                    if self.scenario_predict == "intersection":
                        figure_name = "intersection_" + file + "_plot " + str(i) + ".png"
                    elif self.scenario_predict == "roundabout":
                        figure_name = "roundabout_" + file + "_plot " + str(i) + ".png"
                    file_path = "/Users/anjianli/Desktop/robotics/project/optimized_dp/result/poly_{:d}/{:d}_timesteps/predict_mode/".format(
                        ProcessPredictionV3().degree, ProcessPredictionV3().mode_time_span)
                    # file_path = "/Users/anjianli/Desktop/robotics/project/optimized_dp/result/poly_3/5_timesteps/predict_mode/"
                    figure_path_name = file_path + figure_name
                    # print(figure_path_name)
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    plt.savefig(figure_path_name)

    def get_predict_traj(self, scenario, traj_file_pred=None):

        if traj_file_pred is None:
            # If not specified, then randomly pick a file
            random.seed(13)
            index = random.randint(0, len(self.file_name_predict) - 1)
            traj_file_name = self.file_dir_predict + '/' + self.file_name_predict[index]
            # print("the traj file to predict is", traj_file_name)

            traj_file = ProcessPredictionV3().read_prediction(file_name=traj_file_name)
        else:
            traj_file_name = self.file_dir_predict + '/' + traj_file_pred
            # print("the traj file to predict is", traj_file_name)

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
            acc_interpolate, omega_interpolate = ProcessPredictionV3().to_interpolate(raw_acc_list[i], raw_omega_list[i], mode="prediction")
            # print("acc size", np.shape(acc_interpolate)[0])
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

    def plot_mode(self, mode_num_seq, mode_num_str, acc, omega):

        fig = plt.figure()
        ax1 = fig.add_subplot(311)

        time_index = np.linspace(0, np.shape(mode_num_seq)[0], num=np.shape(mode_num_seq)[0])
        # print(mode_num_seq)
        # print(time_index)

        ax1.plot(time_index, mode_num_seq, 'o-')
        ax1.grid()
        ax1.set_ylabel('mode')
        ax1.set_xlabel('timestep')
        ax1.set_title('0: decelerate, 1: stable, 2: accelerate, 3: left turn, 4: right turn, 5: curve path, -1: other')
        # ax1.set_title('0: decelerate, 1: accelerate, 2: stable, 3: left turn, 4: right turn, -1: other')
        # ax1.set_title('0: stable, 1: decelerate, 2: accelerate, 3: right turn, 4: left turn, -1: other')
        # ax1.set_title('0: right turn, 1: stable, 2: decelerate, 3: left turn, 4: accelerate, -1: other')

        locs, labels = plt.xticks()
        plt.xticks(np.arange(0, np.shape(mode_num_seq)[0], step=ProcessPredictionV3().mode_time_span))
        plt.yticks(np.arange(-1, ClusteringV3().clustering_num, step=1))

        ax2 = fig.add_subplot(312, sharex=ax1)
        ax2.plot(time_index, acc, 'o-')
        ax2.set_ylabel('acceleration')
        ax2.set_xlabel('physical bound [-5, 3]')
        # plt.yticks(np.arange(-5, 3, step=1))

        ax3 = fig.add_subplot(313, sharex=ax1)
        ax3.plot(time_index, omega, 'o-', label="angular speed")
        ax3.set_ylabel('angular speed')
        label_name = "bound: acc:[-5, 3], ang_v: [-pi/6, pi/6]" + str(ProcessPredictionV3().mode_time_span) + "time span"
        ax3.set_xlabel(label_name)
        # plt.yticks(np.arange(-math.pi/6, math.pi/6, step=math.pi/15))

        if not self.to_save_pred_mode:
            plt.show()

if __name__ == "__main__":
    PredictModeV3().predict_mode()
