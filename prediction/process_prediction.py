import numpy as np
import pandas
import csv


class ProcessPrediction(object):

    def __init__(self):

        self.file_dir = '/home/anjianl/Desktop/project/optimized_dp/data/csv_files_for_planner/intersection'
        self.file_name = 'car_16_vid_09.csv'
        # self.file_name = 'car_20_vid_09.csv'

        self.time_step = 1

    def read_prediction(self):
        """
        Read CSV file that contains predicted goal states and generated trajectory

        """

        prediction_file = pandas.read_csv(self.file_dir + '/' + self.file_name)

        # print(prediction_file)
        #
        # print(prediction_file['v_t'])


        return prediction_file

    def get_action_data(self):
        """
        Extract acceleration and orientation speed for each generated trajectory

        """

        traj_file = self.read_prediction()

        length = len(traj_file)

        # Initalize for acceleration
        acceleration_list = []
        linear_velocity = []
        acceleration = []

        # Initialize for angular velocity
        angular_velociy_list = []
        angular_velociy = []
        dx = []
        dy = []
        orientation = []

        # Extract acceleration in the traj_file
        for i in range(length):
            linear_velocity.append(traj_file['v_t'][i])
            if i > 0 and traj_file['t_to_goal'][i - 1] != 0:
                acceleration.append((linear_velocity[i] - linear_velocity[i - 1]) / self.time_step)
            if i > 0 and traj_file['t_to_goal'][i] == 0 and traj_file['t_to_goal'][i - 1] != 0:
                acceleration_list.append(np.asarray(acceleration))
                acceleration = []

        for i in range(length):
            if i > 1 and traj_file['t_to_goal'][i - 1] != 0 and traj_file['t_to_goal'][i - 2] != 0:
                orientation_1 = np.arctan2(traj_file['y_t'][i - 1] - traj_file['y_t'][i - 2], traj_file['x_t'][i - 1] -
                                           traj_file['x_t'][i - 2])
                orientation_2 = np.arctan2(traj_file['y_t'][i] - traj_file['y_t'][i - 1], traj_file['x_t'][i] -
                                           traj_file['x_t'][i - 1])
                angular_velociy.append((orientation_2 - orientation_1) / self.time_step)
            if i > 0 and traj_file['t_to_goal'][i] == 0 and traj_file['t_to_goal'][i - 1] != 0:
                angular_velociy_list.append(np.asarray(angular_velociy))
                angular_velociy = []


        return acceleration_list, angular_velociy_list

    def get_action_bound(self):

        # These are list of numpy arrays
        acceleration_data, angular_velocity_data = self.get_action_data()

        # For acceleration
        upper_acceleration = []
        lower_acceleration = []
        for acceleration in acceleration_data:
            # print(acceleration)
            upper_acceleration.append(np.max(acceleration))
            lower_acceleration.append(np.min(acceleration))

        print('upper bound for acceleration is', upper_acceleration)
        print('lower bound for acceleration is', lower_acceleration)

        # For angular velocity
        upper_ang_v = []
        lower_ang_v = []
        for ang_v in angular_velocity_data:
            # print(ang_v)
            upper_ang_v.append(np.max(ang_v))
            lower_ang_v.append(np.min(ang_v))

        print('upper bound for angular velocity is', upper_ang_v)
        print('lower bound for angular velocity is', lower_ang_v)

        return upper_acceleration, lower_acceleration, upper_ang_v, lower_ang_v

if __name__ == "__main__":

    ProcessPrediction().get_action_bound()