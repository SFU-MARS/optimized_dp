import numpy as np
import pandas
import csv


class ProcessPrediction(object):

    def __init__(self):

        self.file_dir_intersection = '/home/anjianl/Desktop/project/optimized_dp/data/csv_files_for_planner/intersection'
        self.file_name_intersection = ['car_16_vid_09.csv', 'car_20_vid_09.csv', 'car_29_vid_09.csv',
                                       'car_36_vid_11.csv', 'car_50_vid_03.csv', 'car_112_vid_11.csv',
                                       'car_122_vid_11.csv']
        self.file_dir_roundabout = '/home/anjianl/Desktop/project/optimized_dp/data/csv_files_for_planner/roundabout'
        self.file_name_roundabout = ['car_27.csv', 'car_122.csv']

        self.time_step = 0.01
        self.time_filter = 10

    def read_prediction(self, file_name=None):
        """
        Read CSV file that contains predicted goal states and generated trajectory

        """

        # prediction_file = pandas.read_csv(self.file_dir + '/' + self.file_name)
        prediction_file = pandas.read_csv(file_name)

        # print(prediction_file)
        #
        # print(prediction_file['v_t'])


        return prediction_file

    def get_action_data(self, file_name=None):
        """
        Extract acceleration and orientation speed for each generated trajectory

        """

        traj_file = self.read_prediction(file_name=file_name)

        length = len(traj_file)

        # Initalize for acceleration
        acceleration_list = []
        acceleration = []
        acc_tmp = []

        # Initialize for angular velocity
        angular_velociy_list = []
        angular_velociy = []
        ang_v_tmp = []

        num = 1
        # Extract acceleration in the traj_file
        for i in range(length):
            if i > 0 and traj_file['t_to_goal'][i - 1] != 0:
                # Average over some horizon
                if (traj_file['v_t'][i] - traj_file['v_t'][i - 1]) / self.time_step < - 10:
                    print(traj_file['v_t'][i], traj_file['v_t'][i - 1])
                acc_tmp.append((traj_file['v_t'][i] - traj_file['v_t'][i - 1]) / self.time_step)
                if num % self.time_filter == 0:
                    # Mean
                    acceleration.append(np.mean(acc_tmp))
                    # Median
                    # acceleration.append(np.median(acc_tmp))
                    acc_tmp = []
                elif traj_file['t_to_goal'][i] == 0:
                    # Mean
                    acceleration.append(np.mean(acc_tmp))
                    # Median
                    # acceleration.append(np.median(acc_tmp))
                    acc_tmp = []
                num += 1
            if traj_file['t_to_goal'][i] == 0:
                acceleration_list.append(np.asarray(acceleration))
                acceleration = []

        num = 1
        traj_num = 1
        # Time interval = 1 time_step
        for i in range(length):
            if i > 1 and traj_file['t_to_goal'][i - 1] != 0 and traj_file['t_to_goal'][i - 2] != 0:
                orientation_1 = np.arctan2(traj_file['y_t'][i - 1] - traj_file['y_t'][i - 2], traj_file['x_t'][i - 1] -
                                           traj_file['x_t'][i - 2])
                orientation_2 = np.arctan2(traj_file['y_t'][i] - traj_file['y_t'][i - 1], traj_file['x_t'][i] -
                                           traj_file['x_t'][i - 1])
                # Calculate signed angle difference
                angle_difference = self.get_ang_diff(orientation_2, orientation_1)
                ang_v_tmp.append(angle_difference / self.time_step)
                # Average over some time horizon
                if num % self.time_filter == 0:
                    # Mean
                    angular_velociy.append(np.mean(ang_v_tmp))
                    # Median
                    # angular_velociy.append(np.median(ang_v_tmp))
                    ang_v_tmp = []
                elif traj_file['t_to_goal'][i] == 0:
                    # Mean
                    angular_velociy.append(np.mean(ang_v_tmp))
                    # Median
                    # angular_velociy.append(np.median(ang_v_tmp))
                    ang_v_tmp = []
                num += 1
                # if np.abs(angle_difference / self.time_step) > 1:
                #     print(traj_file['t'][i], traj_num, 'base is %.3f' % orientation_1, 'target is %.3f' % orientation_2, 'a_v is %.2f' % (angle_difference / self.time_step))
            if traj_file['t_to_goal'][i] == 0:
                angular_velociy_list.append(np.asarray(angular_velociy))
                angular_velociy = []
                traj_num += 1

        return acceleration_list, angular_velociy_list

    def get_action_bound(self, file_name=None):

        # These are list of numpy arrays
        acceleration_data, angular_velocity_data = self.get_action_data(file_name=file_name)

        # For acceleration
        upper_acceleration = []
        lower_acceleration = []
        for acceleration in acceleration_data:
            # print(acceleration)
            upper_acceleration.append(np.max(acceleration))
            lower_acceleration.append(np.min(acceleration))

        # print('upper bound for acceleration is', upper_acceleration)
        # print('lower bound for acceleration is', lower_acceleration)

        # For angular velocity
        upper_ang_v = []
        lower_ang_v = []
        for ang_v in angular_velocity_data:
            # print(ang_v)
            upper_ang_v.append(np.max(ang_v))
            lower_ang_v.append(np.min(ang_v))

        # print('upper bound for angular velocity is', upper_ang_v)
        # print('lower bound for angular velocity is', lower_ang_v)

        return upper_acceleration, lower_acceleration, upper_ang_v, lower_ang_v

    def collect_action_bound_from_group(self):

        # in the format of: (file, a_upper, a_lower, ang_v_upper, ang_v_lower)
        action_bound_list_intersection = []
        action_bound_list_roundabout = []

        # For intersection scenario
        for file_name in self.file_name_intersection:
            full_file_name = self.file_dir_intersection + '/' + file_name
            a_upper, a_lower, ang_v_upper, ang_v_lower = self.get_action_bound(file_name=full_file_name)
            for index in range(len(a_upper)):
                action_bound_list_intersection.append([file_name, a_upper[index], a_lower[index], ang_v_upper[index], ang_v_lower[index]])

        # For roundabout scenario
        for file_name in self.file_name_roundabout:
            full_file_name = self.file_dir_roundabout + '/' + file_name
            a_upper, a_lower, ang_v_upper, ang_v_lower = self.get_action_bound(file_name=full_file_name)
            for index in range(len(a_upper)):
                action_bound_list_roundabout.append([file_name, a_upper[index], a_lower[index], ang_v_upper[index], ang_v_lower[index]])

        print("intersection")
        print(action_bound_list_intersection)
        print("roundabout")
        print(action_bound_list_roundabout)

        return action_bound_list_intersection, action_bound_list_roundabout

    def get_ang_diff(self, target, base):
        """
        Given target and base both in [-pi, pi], calculate signed angle difference of (target - base)

        """

        # If either target or base == 0, then return 0
        if target == 0 or base == 0:
            return 0
        elif target > base:
            if target - base <= np.pi:
                return target - base
            else:
                return target - (base + 2 * np.pi)
        elif target == base:
            return 0
        else:
            return - self.get_ang_diff(base, target)

        # # Doesn't consider target or base == 0
        # if target > base:
        #     if target - base <= np.pi:
        #         return target - base
        #     else:
        #         return target - (base + 2 * np.pi)
        # elif target == base:
        #     return 0
        # else:
        #     return - self.get_ang_diff(base, target)

if __name__ == "__main__":
    ProcessPrediction().collect_action_bound_from_group()