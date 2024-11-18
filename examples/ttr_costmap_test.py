import argparse

import numpy as np
import matplotlib as mpl
import time
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from odp.Shapes import *
from odp.dynamics import DubinsCar4D, DubinsCar


from robot_utils.ttr_solver import TTRSolver


def plot_contour(costmap, costmap_info, V_slice, selected_slice, traj=None):
    # Create the contour plot
    plt.figure(figsize=(12, 12))
    x0 = costmap_info['origin'][0]
    y0 = costmap_info['origin'][1]
    dx = costmap_info['resolution']
    dy = costmap_info['resolution']
    Nx = costmap_info['width']
    Ny = costmap_info['height']
    xn = x0 + dx*Nx
    yn = y0 + dy*Ny
    # plot the map
    xm = np.arange(x0, xn, dx)
    ym = np.arange(y0, yn, dy)
    Xm, Ym = np.meshgrid(xm, ym, indexing='xy')
    plt.pcolor(Xm, Ym, costmap)
    plt.xlim([-4, 3])
    plt.ylim([4, 12])
    # plot the TTR value function
    xv = np.linspace(x0, xn, int(Nx))
    yv = np.linspace(y0, yn, int(Ny))
    Xv, Yv = np.meshgrid(xv, yv)
    levels = np.arange(np.floor(V_slice.min()), 50, 1)  # np.ceil(V_slice.max())+1
    contour = plt.contour(Xv, Yv, V_slice.T, levels=levels, cmap='rainbow', linewidths=2.0)
    plt.clabel(contour, inline=True, fontsize=20, fmt='%d')
    plt.colorbar(contour)
    # Trajectory
    if traj is not None:
        x_traj = traj[:, 0]    
        y_traj = traj[:, 1]
        plt.plot(x_traj[0], y_traj[0], color='blue', marker='P', markersize=10, label='start point')
        plt.plot(x_traj[1:], y_traj[1:], color='red', marker='*', label='Trajectory')
    # plt.pcolor(X, Y, costmap)
    plt.title(f'Contour map at heading_angle slice {selected_slice}', fontsize=20)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    
def check_theta(angle):
    # Make sure the angle is in the range of [0, 2*pi)
    while angle >=2*np.pi:
        angle -= 2 * np.pi
    while angle < 0:
        angle += 2 * np.pi

    return angle

def deg2radian(state):
    _, _, degree = state
    radian = degree / 180*np.pi
    radian = check_theta(radian)
    state[2] = radian
    return state


def main():
    # LOAD data
    npzfile = np.load("debug_data/a_star_search_input_data.npz")
    costmap = npzfile["global_costmap"]  # shape (height, width)
    costmap_info = npzfile["costmapinfo"]
    costmap_info = {
        "height": costmap_info[0],  
        "width": costmap_info[1],  
        "resolution": costmap_info[2],
        "origin": [costmap_info[3], costmap_info[4]],
        }
    pose_list= [[-2.5, 5.2, -90.0], [1.0, 5.2, 0.0], [-2.6, 5.2, 180.0], [-2.6, 10.5, 90.0]] # Big L
    
    # Choose the dynamics
    dyn = DubinsCar(uMode="min")
    # dyn = DubinsCar4D(uMode="min", dMin=[0.0, 0.0], dMax=[0.0, 0.0])
    ttr_solver = TTRSolver(dyn=dyn, costmap=costmap, costmap_info=costmap_info)
    goal= pose_list[0]
    goal = deg2radian(goal)
    ttr = ttr_solver.computeTTR(goal, plot=False)
    
    # Traj test
    start_state = np.array([1.5, 5.2, 3.14]) 
    ctrl_freq = 5
    selected_slice = int(start_state[2]/(2*np.pi)*36)
    # ttr_clip = np.clip(ttr, -1, 200)
    ttr_slice = ttr[:, :, selected_slice]

    # from scipy.interpolate import RegularGridInterpolator
    # xmin = costmap_info['origin'][0]
    # ymin = costmap_info['origin'][1]
    # dx = costmap_info['resolution']
    # dy = costmap_info['resolution']
    # Nx = costmap_info['width']
    # Ny = costmap_info['height']
    # xmax = xmin + dx*Nx
    # ymax = ymin + dy*Ny
    # theta_min, theta_max = 0.0, 2 * np.pi  # Replace with actual range for the third dimension
    # # Generate the grid (replace linspace values if necessary)
    # x = np.linspace(xmin, xmax, int(Nx))
    # y = np.linspace(ymin, ymax, int(Ny))
    # theta = np.linspace(theta_min, theta_max, 36)
    # interpolated_function = RegularGridInterpolator((x, y, theta), ttr)
    # interpolated_value = interpolated_function(start_state)
    # print("Interpolated value:", interpolated_value)
    start_time = time.time()
    traj = ttr_solver.generate_trajectory(ttr, start_state, ctrl_freq, goal)
    end_time = time.time()
    print(f"The trajectory generation takes {end_time-start_time} seconds.")

    plot_contour(costmap, costmap_info, ttr_slice, selected_slice, traj) 


if __name__ == "__main__":
    main()