import numpy as np
import pandas
import math
import matplotlib.pyplot as plt
import csv
import copy


class ProcessPredictionV3(object):

    def __init__(self):

        # Choose which scenario to use for clustering
        self.scenario_to_use = ["intersection", "roundabout"]
        # self.scenario_to_use = ["intersection"]
        # self.scenario_to_use = ["roundabout"]

        # Data directory
        # Remote desktop
        self.file_dir_intersection = '/home/anjianl/Desktop/project/optimized_dp/data/intersection-data'
        self.file_dir_roundabout = '/home/anjianl/Desktop/project/optimized_dp/data/roundabout-data'
        # My laptop
        # self.file_dir_intersection = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/intersection-data'
        # self.file_dir_roundabout = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/roundabout-data'

        # File name
        self.file_name_intersection = ['car_16_vid_09.csv', 'car_20_vid_09.csv', 'car_29_vid_09.csv',
                                       'car_36_vid_11.csv', 'car_50_vid_03.csv', 'car_112_vid_11.csv',
                                       'car_122_vid_11.csv',
                                       'car_38_vid_02.csv', 'car_52_vid_07.csv', 'car_73_vid_02.csv',
                                       'car_118_vid_11.csv']
        self.file_name_roundabout = ['car_27.csv', 'car_122.csv',
                                     'car_51.csv', 'car_52.csv', 'car_131.csv', 'car_155.csv']

        # Time step
        self.time_step = 0.1
        self.time_filter = 10

        # Within a time span, we only specify single mode for the trajectory
        # self.mode_time_span = 20
        self.mode_time_span = 10

        # Action bound, used for filtering
        self.acc_bound = [-5, 3]
        self.omega_bound = [- math.pi / 6, math.pi / 6]

        # Fit polynomial
        self.degree = 5

        # Use velocity profile or not
        self.use_velocity = True
        # self.use_velocity = False

        # For outliers, we choose whether to interpolate and replace the outliers
        # The interpolation is: if the outliers is sandwiched by 2 normal value, then replace it with the mean of the former and latter
        self.to_interpolate_outlier = True
        # self.to_interpolate_outlier = False

    def get_action_data(self, file_name=None):
        """
        Extract acceleration and orientation speed for predicted trajectory

        """

        traj_file = self.read_prediction(file_name=file_name)

        # Extract trajectory segments from csv files
        raw_traj = self.extract_traj(traj_file)

        # Fit polynomial for x, y position: x(t), y(t)
        poly_traj = self.fit_polynomial_traj(raw_traj)

        if self.use_velocity:
            # Get the acc from velocity profile provided
            raw_acc_list, raw_omega_list = self.get_action_v_profile(raw_traj, poly_traj)
        else:
            # Get raw actions from poly_traj, here acc and omega are extracted from both poly_traj
            raw_acc_list, raw_omega_list = self.get_action_poly(poly_traj)

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
        traj_seg['v_t'] = []

        for i in range(length):
            traj_seg['x_t'].append(traj_file['x_t'][i])
            traj_seg['y_t'].append(traj_file['y_t'][i])
            traj_seg['v_t'].append(traj_file['v_t'][i])
            if traj_file['t_to_goal'][i] == 0:
                raw_traj.append(traj_seg)
                traj_seg = {}
                traj_seg['x_t'] = []
                traj_seg['y_t'] = []
                traj_seg['v_t'] = []

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

    def get_action_poly(self, poly_traj):
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

            # np.set_printoptions(precision=2)
            # print("omega is", omega_t)
            # print("min", np.min(omega_t),"max", np.max(omega_t))

            if len(a_t) != len(omega_t):
                print("a_t and omega_t have different dimensions")
                raise SystemExit(0)

        return acceleration_list, omega_list

    def get_action_v_profile(self, raw_traj, poly_traj):

        acceleration_list = []
        omega_list = []

        # Only use omega data from poly_traj
        _, omega_list = self.get_action_poly(poly_traj)

        for raw_traj_seg in raw_traj:
            length = len(raw_traj_seg['v_t'])
            v_t = np.asarray(raw_traj_seg['v_t'])

            a_t = (v_t[2:length] - v_t[0:(length - 2)]) / (2 * self.time_step)
            acceleration_list.append(a_t)

            # np.set_printoptions(precision=2)
            # print("acc is", a_t)
            # print("min", np.min(a_t),"max", np.max(a_t))

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

        num_all = 0
        num_effective = 0
        for i in range(episode_num):
            episode_len = np.shape(raw_acc_list[i])[0]
            # If the episode length is less than time span, filter it out
            if episode_len < self.mode_time_span:
                continue

            # Interpolate and replace the outlier for acc and omega
            if self.to_interpolate_outlier:
                acc_interpolate, omega_interpolate = self.to_interpolate(raw_acc_list[i], raw_omega_list[i])
                raw_acc_list[i] = acc_interpolate
                raw_omega_list[i] = omega_interpolate

            for j in range(episode_len):
                if j + self.mode_time_span <= episode_len:
                    num_all += 1

                    acc_span = raw_acc_list[i][j:j + self.mode_time_span]
                    omega_span = raw_omega_list[i][j:j + self.mode_time_span]

                    # If in the current timespan, there's a time step that is outside the action bound, filter out the this time span
                    # So we only leave those timespan that actions are within bound for all timestep
                    if (np.min(acc_span) >= self.acc_bound[0]) and (np.max(acc_span) <= self.acc_bound[1]) and \
                            (np.min(omega_span) >= self.omega_bound[0]) and (np.max(omega_span) <= self.omega_bound[1]):
                        num_effective += 1

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

        # print("total number is", num_all, "effective number is", num_effective, "the ratio is {:.2f}".format(num_effective / num_all))

        return filtered_acc_mean_list, filtered_acc_variance_list, filtered_omega_mean_list, filtered_omega_variance_list

    def to_interpolate(self, acc, omega):

        len = np.shape(acc)[0]

        for i in range(len):
            if i != 0 and i != (len - 1):
                if (not self.acc_in_bound(acc[i])) and (self.acc_in_bound(acc[i - 1])) and (self.acc_in_bound(acc[i + 1])):
                    acc[i] = (acc[i - 1] + acc[i + 1]) / 2
                if (not self.omega_in_bound(omega[i])) and (self.omega_in_bound(omega[i - 1])) and (self.omega_in_bound(omega[i + 1])):
                    omega[i] = (omega[i - 1] + omega[i + 1]) / 2

        return acc, omega

    def acc_in_bound(self, acc):

        if acc < self.acc_bound[0] or acc > self.acc_bound[1]:
            return False
        else:
            return True

    def omega_in_bound(self,omega):

        if omega < self.omega_bound[0] or omega > self.omega_bound[1]:
            return False
        else:
            return True

    def collect_action_from_group(self):

        # in the format of: (file, a_upper, a_lower, ang_v_upper, ang_v_lower)
        filename_action_feature_list = []
        filename_list = []

        # Use both intersection and roundabout
        for scenario in self.scenario_to_use:
            if scenario == "intersection":
                filename_list = copy.copy(self.file_name_intersection)
                file_dir = copy.copy(self.file_dir_intersection)
            elif scenario == "roundabout":
                filename_list = copy.copy(self.file_name_roundabout)
                file_dir = copy.copy(self.file_dir_roundabout)
            for file_name in filename_list:
                full_file_name = file_dir + '/' + file_name
                acc_mean, acc_variance, omega_mean, omega_variance = self.get_action_data(file_name=full_file_name)
                for index in range(len(acc_mean)):
                    filename_action_feature_list.append(
                        [file_name, acc_mean[index], acc_variance[index], omega_mean[index], omega_variance[index]])

        # # Only use intersection scenario
        # if self.scenario_to_use == "intersection":
        #     for file_name in self.file_name_intersection:
        #         full_file_name = self.file_dir_intersection + '/' + file_name
        #         acc_mean, acc_variance, omega_mean, omega_variance = self.get_action_data(file_name=full_file_name)
        #         for index in range(len(acc_mean)):
        #             filename_action_feature_list.append([file_name, acc_mean[index], acc_variance[index], omega_mean[index], omega_variance[index]])
        #
        # # Only use roundabout scenario
        # if self.scenario_to_use == "roundabout":
        #     for file_name in self.file_name_roundabout:
        #         full_file_name = self.file_dir_roundabout + '/' + file_name
        #         acc_mean, acc_variance, omega_mean, omega_variance = self.get_action_data(file_name=full_file_name)
        #         for index in range(len(acc_mean)):
        #             filename_action_feature_list.append(
        #                 [file_name, acc_mean[index], acc_variance[index], omega_mean[index], omega_variance[index]])

        return filename_action_feature_list

if __name__ == "__main__":
    ProcessPredictionV3().collect_action_from_group()
