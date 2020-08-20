import numpy as np
import random

from prediction.clustering_v3 import ClusteringV3
from prediction.process_prediction_v3 import ProcessPredictionV3

class PredictModeV3(object):

    def __init__(self):

        # Which scenario to predict
        self.scenario_predict = "intersection"

        # Data directory
        self.file_dir_intersection = '/home/anjianl/Desktop/project/optimized_dp/data/intersection-data'
        self.file_dir_roundabout = '/home/anjianl/Desktop/project/optimized_dp/data/roundabout-data'

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

        raw_acc_list, raw_omega_list = self.get_predict_traj(scenario=self.scenario_predict)

        raw_acc, raw_omega = raw_acc_list[0], raw_omega_list[0]

        print(self.action_bound_mode)

        for i in range(np.shape(raw_acc)[0]):
            if i + ProcessPredictionV3().mode_time_span < np.shape(raw_acc)[0]:
                curr_mode = self.decide_mode(raw_acc[i:i + ProcessPredictionV3().mode_time_span], raw_omega[i:i + ProcessPredictionV3().mode_time_span])
                print(curr_mode)

        return 0

    def get_predict_traj(self, scenario):

        if scenario == "intersection":
            index = random.randint(0, len(self.file_name_intersection) - 1)
            traj_file_name = self.file_dir_intersection + '/' + self.file_name_intersection[index]

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

    def decide_mode(self, acc, omega):

        if (np.shape(acc)[0] != ProcessPredictionV3().mode_time_span) or (np.shape(omega)[0] != ProcessPredictionV3().mode_time_span):
            print("prediction dimension is wrong")
            return 0

        acc_mean = np.mean(acc)
        omega_mean = np.mean(omega)

        for i in range(len(self.action_bound_mode)):
            if (self.action_bound_mode[i][1] <= acc_mean <= self.action_bound_mode[i][2]) and (self.action_bound_mode[i][3] <= omega_mean <= self.action_bound_mode[i][4]):
                return self.action_bound_mode[i][0]

        return "mode -1"

if __name__ == "__main__":
    PredictModeV3().predict_mode()