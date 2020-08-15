import numpy as np
import pandas
import math
import matplotlib.pyplot as plt
import csv


class ProcessPredictionV3(object):

    def __init__(self):
        # Remote desktop
        self.file_dir_intersection = '/home/anjianl/Desktop/project/optimized_dp/data/intersection-data'
        self.file_dir_roundabout = '/home/anjianl/Desktop/project/optimized_dp/data/csv_files_for_planner/roundabout'
        # My laptop
        # self.file_dir_intersection = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/intersection-data'
        # self.file_dir_roundabout = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/csv_files_for_planner/roundabout'

        # File name
        self.file_name_intersection = ['car_16_vid_09.csv', 'car_20_vid_09.csv', 'car_29_vid_09.csv',
                                       'car_36_vid_11.csv', 'car_50_vid_03.csv', 'car_112_vid_11.csv',
                                       'car_122_vid_11.csv',
                                       'car_38_vid_02.csv', 'car_52_vid_07.csv', 'car_73_vid_02.csv',
                                       'car_118_vid_11.csv']
        self.file_name_roundabout = ['car_27.csv', 'car_122.csv']

        # Time step
        self.time_step = 0.1
        self.time_filter = 10

        # Within a time span, we only specify single mode for the trajectory
        self.mode_time_span = 20

        # Action bound, used for filtering
        self.acc_bound = [-5, 3]
        self.omega_bound = [- math.pi / 6, math.pi / 6]

        # Fit polynomial
        self.degree = 5

    def get_action_data(self, file_name=None):
        """
        Extract acceleration and orientation speed for predicted trajectory

        """

        traj_file = self.read_prediction(file_name=file_name)

        # Extract trajectory segments from csv files
        raw_traj = self.extract_traj(traj_file)

        # Fit polynomial for x, y position: x(t), y(t)
        poly_traj = self.fit_polynomial_traj(raw_traj)

        # Get raw actions from poly_traj
        raw_acc_list, raw_omega_list = self.get_action(poly_traj)

        # Filter out short episode, outliers (if the timestep has outliers), and get mean and variance over some
        # time-span of mode
        filtered_acc_mean_list, filtered_acc_variance_list, filtered_omega_mean_list, filtered_omega_variance_list = \
            self.get_mean_variance(raw_acc_list, raw_omega_list)

        return filtered_acc_mean_list, filtered_acc_variance_list, filtered_omega_mean_list, filtered_omega_variance_list

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

    def extract_traj(self, traj_file):
        """
        Extract trajectory segments from csv file

        :param traj_file:
        :return:
        """
        length = len(traj_file)

        raw_traj = []
        traj_seg = {}
        traj_seg['x_t'] = []
        traj_seg['y_t'] = []

        for i in range(length):
            traj_seg['x_t'].append(traj_file['x_t'][i])
            traj_seg['y_t'].append(traj_file['y_t'][i])
            if traj_file['t_to_goal'][i] == 0:
                raw_traj.append(traj_seg)
                traj_seg = {}
                traj_seg['x_t'] = []
                traj_seg['y_t'] = []

        return raw_traj

    def fit_polynomial_traj(self, raw_traj):
        """
        Given a trajectory segment, fit a polynomial
        :param raw_traj:
        :return:
        """

        poly_traj = []

        for raw_traj_seg in raw_traj:
            t = np.asarray(range(len(raw_traj_seg['x_t'])))
            x_t = np.asarray(raw_traj_seg['x_t'])
            y_t = np.asarray(raw_traj_seg['y_t'])

            # Fit a 5-degree polynomial
            degree = self.degree
            weights_x = np.polyfit(t, x_t, degree)
            weights_y = np.polyfit(t, y_t, degree)

            poly_x_func = np.poly1d(weights_x)
            poly_x_t = np.asarray([poly_x_func(t) for t in range(len(t))])
            poly_y_func = np.poly1d(weights_y)
            poly_y_t = np.asarray([poly_y_func(t) for t in range(len(t))])

            # Plot the comparison
            # plt.plot(t, y_t)
            # plt.plot(t, poly_y_t)
            # plt.show()

            # Form the polynomial trajectory list
            poly_traj_seg = {}
            poly_traj_seg['x_t_poly'] = poly_x_t
            poly_traj_seg['y_t_poly'] = poly_y_t
            poly_traj.append(poly_traj_seg)

        return poly_traj

    def get_action(self, poly_traj):
        """
        Extract acceleration and omega (angular speed) for each generated trajectory

        """

        acceleration_list = []
        omega_list = []

        for poly_traj_seg in poly_traj:

            length = len(poly_traj_seg['x_t_poly'])
            # Directly compute dx/dt, dy/dt
            dx_t = (poly_traj_seg['x_t_poly'][1:length] - poly_traj_seg['x_t_poly'][0:(length - 1)]) / self.time_step
            dy_t = (poly_traj_seg['y_t_poly'][1:length] - poly_traj_seg['y_t_poly'][0:(length - 1)]) / self.time_step

            dx1_t = (poly_traj_seg['x_t_poly'][1:(length - 1)] - poly_traj_seg['x_t_poly'][0:(length - 2)]) / self.time_step
            dx2_t = (poly_traj_seg['x_t_poly'][2:length] - poly_traj_seg['x_t_poly'][1:(length - 1)]) / self.time_step
            dy1_t = (poly_traj_seg['y_t_poly'][1:(length - 1)] - poly_traj_seg['y_t_poly'][0:(length - 2)]) / self.time_step
            dy2_t = (poly_traj_seg['y_t_poly'][2:length] - poly_traj_seg['y_t_poly'][1:(length - 1)]) / self.time_step

            # Finite difference for derivative of x and y
            # dx_t_fd, dy_t_fd = self.get_dx_dy_finite_difference(poly_traj_seg['x_t_poly'], poly_traj_seg['y_t_poly'])
            # dx1_t_fd, dy1_t_fd = self.get_dx_dy_finite_difference(poly_traj_seg['x_t_poly'][0:-1], poly_traj_seg['y_t_poly'][0:-1])
            # dx2_t_fd, dy2_t_fd = self.get_dx_dy_finite_difference(poly_traj_seg['x_t_poly'][1:], poly_traj_seg['y_t_poly'][1:])

            # Get acceleration
            v_t = np.sqrt(dx_t ** 2 + dy_t ** 2)
            # v_t = np.sqrt(dx_t_fd ** 2 + dy_t_fd ** 2)

            a_t = (v_t[1:len(v_t)] - v_t[0:(len(v_t) - 1)]) / self.time_step
            acceleration_list.append(a_t)

            # Get omega (angular speed)
            orientation_1 = np.arctan2(dy1_t, dx1_t)
            orientation_2 = np.arctan2(dy2_t, dx2_t)
            # orientation_1 = np.arctan2(dy1_t_fd, dx1_t_fd)
            # orientation_2 = np.arctan2(dy2_t_fd, dx2_t_fd)

            omega_t = (orientation_2 - orientation_1) / self.time_step
            omega_list.append(omega_t)

            if len(a_t) != len(omega_t):
                print("a_t and omega_t have different dimensions")
                raise SystemExit(0)

        return acceleration_list, omega_list

    def get_dx_dy_finite_difference(self, traj_x, traj_y):

        length = len(traj_x)
        x_t_1 = traj_x[0:(length - 2)]
        x_t_2 = traj_x[2:length]

        y_t_1 = traj_y[0:(length - 2)]
        y_t_2 = traj_y[2:length]

        dx_t_fd = (x_t_2 - x_t_1) / self.time_step
        dy_t_fd = (y_t_2 - y_t_1) / self.time_step

        return dx_t_fd, dy_t_fd

    def get_mean_variance(self, raw_acc_list, raw_omega_list):

        filtered_acc_mean_list = []
        filtered_acc_variance_list = []
        filtered_omega_mean_list = []
        filtered_omega_variance_list = []

        filtered_acc_mean_tmp = []
        filtered_acc_variance_tmp = []
        filtered_omega_mean_tmp = []
        filtered_omega_variance_tmp = []

        episode_num = len(raw_acc_list)

        for i in range(episode_num):
            episode_len = np.shape(raw_acc_list[i])[0]
            if episode_len < self.mode_time_span:
                continue
            for j in range(episode_len):
                if j + self.mode_time_span <= episode_len:
                    acc_span = raw_acc_list[i][j:j + self.mode_time_span]
                    omega_span = raw_omega_list[i][j:j + self.mode_time_span]
                    # If in the current timespan, there's a time step that is outside the action bound, filter out the this time span
                    if (np.min(acc_span) >= self.acc_bound[0]) and (np.max(acc_span) <= self.acc_bound[1]) and \
                            (np.min(omega_span) >= self.omega_bound[0]) and (np.max(omega_span) <= self.omega_bound[1]):
                        filtered_acc_mean_tmp.append(np.mean(acc_span))
                        filtered_acc_variance_tmp.append(np.var(acc_span))
                        filtered_omega_mean_tmp.append(np.mean(omega_span))
                        filtered_omega_variance_tmp.append(np.var(omega_span))
            filtered_acc_mean_list.append(np.asarray(filtered_acc_mean_tmp))
            filtered_acc_variance_list.append(np.asarray(filtered_acc_variance_tmp))
            filtered_omega_mean_list.append(np.asarray(filtered_omega_mean_tmp))
            filtered_omega_variance_list.append((np.asarray(filtered_omega_variance_tmp)))

            filtered_acc_mean_tmp = []
            filtered_acc_variance_tmp = []
            filtered_omega_mean_tmp = []
            filtered_omega_variance_tmp = []

        return filtered_acc_mean_list, filtered_acc_variance_list, filtered_omega_mean_list, filtered_omega_variance_list

    def collect_action_from_group(self):

        # in the format of: (file, a_upper, a_lower, ang_v_upper, ang_v_lower)
        filename_action_feature_list_intersection = []
        file_name_action_feature_list_roundabout = []

        # For intersection scenario
        for file_name in self.file_name_intersection:
            full_file_name = self.file_dir_intersection + '/' + file_name
            acc_mean, acc_variance, omega_mean, omega_variance = self.get_action_data(file_name=full_file_name)
            for index in range(len(acc_mean)):
                filename_action_feature_list_intersection.append([file_name, acc_mean[index], acc_variance[index], omega_mean[index], omega_variance[index]])

        # For roundabout scenario
        for file_name in self.file_name_roundabout:
            full_file_name = self.file_dir_roundabout + '/' + file_name
            acc_mean, acc_variance, omega_mean, omega_variance = self.get_action_data(file_name=full_file_name)
            for index in range(len(acc_mean)):
                file_name_action_feature_list_roundabout.append(
                    [file_name, acc_mean[index], acc_variance[index], omega_mean[index], omega_variance[index]])

        return filename_action_feature_list_intersection, file_name_action_feature_list_roundabout

if __name__ == "__main__":
    ProcessPredictionV3().collect_action_from_group()
