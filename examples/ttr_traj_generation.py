import argparse
import time
import numpy as np
import os
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from odp.Shapes import *
# from robot_utils.DubinCars import DubinsCar, DubinsCar4D
from odp.dynamics.DubinsCar2 import DubinsCar2
from robot_utils.ttr_planner import TTRSolver


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
    plt.ylim([4, 15])
    # plot the TTR value function
    xv = np.linspace(x0, xn, int(Nx))
    yv = np.linspace(y0, yn, int(Ny))
    Xv, Yv = np.meshgrid(xv, yv)
    levels = np.arange(np.floor(V_slice.min()), 100, 1)  # np.ceil(V_slice.max())+1
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
    # print(f"The shape of the costmap is {costmap.shape}")
    costmap_info = npzfile["costmapinfo"]
    costmap_info = {
        "height": costmap_info[0],  
        "width": costmap_info[1],  
        "resolution": costmap_info[2],
        "origin": [costmap_info[3], costmap_info[4]],
        }
    pose_list= [[-2.5, 5.2, -90.0], [1.0, 5.2, 0.0], [-2.6, 5.2, 180.0], [-2.6, 10.5, 90.0]] # For real-world
    pose_sim_list= [[-2.6, 5.1, -45.0], [0.5, 5.1, 180.0], [-2.6, 5.1, 135.0],  [-2.6, 8.0, -90.0]]  # For simulation
    
    # Choose the dynamics
    uMin=[-0.1, -1.0]
    uMax =[0.8, 1.0]
    solver = "TTR"
    Dynamics = "3D with 2 controls"
    
    dyn = DubinsCar2(uMode="min", uMin=uMin, uMax =uMax)
    ttr_solver = TTRSolver(dyn=dyn, costmap=costmap, costmap_info=costmap_info)
    dyn_dim = ttr_solver.dim
    
    # # Compute TTR value functions
    # # save_path = os.getcwd() + f'/test/TTR3D_Sim'
    # save_path = os.getcwd() + f'/test/TTR3D_World'
    # file_path = save_path + '/configs.txt'
    # with open(file_path, 'w') as file:
    #     file.write(f"uMin = {uMin}\n")
    #     file.write(f"uMax = {uMax}\n")
    #     file.write(f"TTRSolver = {solver}\n")
    #     file.write(f"Dynamics = {Dynamics}")
        
    # for goal in pose_list:
    #     save_name = deg2radian(goal)
    #     # save_name.insert(2, 0.0)  # add one position for goal region generation with DubinCar4D 
    #     goal = np.array(save_name)  # x, y, theta
    #     ttr = ttr_solver.computeTTR(goal, plot=False, method=solver)
    #     np.save(f"{save_path}/goal{save_name}.npy", ttr)
    #     print(f"The goal{save_name}.npy is saved!")
    
    # Load the TTR value function for testing
    goal= pose_sim_list[3]
    load_name = deg2radian(goal)
    if dyn_dim == 3:
        load_path = os.getcwd() +'/test/TTR3D_Sim'
    elif dyn_dim == 4:
        load_path = os.getcwd() +'/test/TTR_values_4D'
    
    load_value = os.path.join(load_path, f'goal{load_name}.npy')
    ttr = np.load(load_value)
    # Traj test
    ctrl_freq = 30
    if dyn_dim == 3:
        start_state = np.array([1.5, 5.2, 3.14]) 
        theta_slice = int(start_state[2]/(2*np.pi)*150)
    elif dyn_dim == 4:
        start_state = np.array([1.5, 5.2, 1.0, 3.14])
        theta_slice = int(start_state[3]/(2*np.pi)*36)
    # ttr_clip = np.clip(ttr, -1, 200)
    if dyn_dim == 3:
        ttr_slice = ttr[:, :, theta_slice]
    elif dyn_dim == 4:
        ttr_slice = ttr[:, :, 18, theta_slice]  # Nv = 20, 0.0~1.1
    
    # Generate the traj
    goal = deg2radian(goal)
    if dyn_dim == 4:
        goal.insert(2, 0.0)  # add one position for goal region generation 
    start_time = time.time()
    # breakpoint()
    traj, controls = ttr_solver.generate_trajectory(ttr, start_state, ctrl_freq, goal)
    end_time = time.time()
    print(f"The trajectory generation takes {end_time-start_time} seconds.")
    plot_contour(costmap, costmap_info, ttr_slice, theta_slice, traj) 


if __name__ == "__main__":
    main()