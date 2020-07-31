import numpy as np
import pickle

from prediction.process_prediction_v2 import *

class PredictMode(object):

    def __init__(self):
        # Remote desktop
        self.file_dir_intersection = '/home/anjianl/Desktop/project/optimized_dp/data/csv_files_for_planner/intersection'
        self.file_dir_roundabout = '/home/anjianl/Desktop/project/optimized_dp/data/csv_files_for_planner/roundabout'
        # My laptop
        # self.file_dir_intersection = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/csv_files_for_planner/intersection'
        # self.file_dir_roundabout = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/csv_files_for_planner/roundabout'

        self.file_name_intersection = ['car_16_vid_09.csv', 'car_20_vid_09.csv', 'car_29_vid_09.csv',
                                       'car_36_vid_11.csv', 'car_50_vid_03.csv', 'car_112_vid_11.csv',
                                       'car_122_vid_11.csv']
        self.file_name_roundabout = ['car_27.csv', 'car_122.csv']

    def get_action_from_prediction(self):

        filename = self.file_dir_intersection + '/' + self.file_name_intersection[0]

        acc_list, omega_list = ProcessPredictionV2().get_action_data(file_name=filename)

        acc_omega_list = []
        acc_omega = []
        num_in_effect = 0

        for i in range(len(acc_list)):
            for j in range(len(acc_list[i])):
                # if (-5 <= acc_list[i][j] <= 3) and (-0.34 <= omega_list[i][j] <= 0.34):
                acc_omega.append([acc_list[i][j], omega_list[i][j]])
                if j == len(acc_list[i]) - 1:
                    acc_omega_list.append(np.asarray(acc_omega))
                    acc_omega = []

        return acc_omega_list

    def get_mode(self):

        # Get actions from predicted trajectory. Might have several sets of actions
        acc_omega_list = self.get_action_from_prediction()

        # load kmeans model
        kmeans_action = pickle.load(open("/home/anjianl/Desktop/project/optimized_dp/model/kmeans_action_intersection.pkl", "rb"))

        for seg in acc_omega_list:
            print(kmeans_action.predict(seg))

        return 0

if __name__ == "__main__":
    PredictMode().get_mode()
