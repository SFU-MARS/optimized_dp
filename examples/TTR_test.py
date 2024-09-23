import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *
# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar, Plane2D, DubinsCar4D
# Plot options
from odp.Plots import PlotOptions 
from odp.Plots.plotting_utilities import plot_isosurface
# Solver core

from odp.solver import HJSolver, TTRSolver
from odp.compute_trajectory_TTR import compute_opt_traj_TTR
import math


def plot_contour(V_slice, selected_slice, xmin, xmax, ymin, ymax, grid1, grid2, traj=None):
    # Create the contour plot
    plt.figure(figsize=(12, 12))
    x_values = np.linspace(xmin, xmax, grid1)
    y_values = np.linspace(ymin, ymax, grid2)
    X, Y  = np.meshgrid(x_values, y_values)
    # Contour
    levels = np.arange(np.floor(V_slice.min()), np.ceil(V_slice.max())+1, 1)
    contour = plt.contour(X, Y, V_slice.T, levels=levels, cmap='rainbow', linewidths=3.0)
    plt.clabel(contour, inline=True, fontsize=20, fmt='%d')
    # Trajectory
    if traj is not None:
        x_traj = traj[:, 0]    
        y_traj = traj[:, 1]
        plt.plot(x_traj[0], y_traj[0], color='blue', marker='P', markersize=10, label='start point')
        plt.plot(x_traj[1:], y_traj[1:], color='red', marker='*', label='Trajectory')

    plt.title(f'Contour map at heading_angle slice {selected_slice}', fontsize=20)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


# -------------------------------- ONE-SHOT TTR COMPUTATION ---------------------------------- #

# TTR Example 1
g = Grid(minBounds=np.array([-3.0, -1.0, -math.pi]), maxBounds=np.array([3.0, 4.0, math.pi]),
         dims=3, pts_each_dim=np.array([50, 50, 50]), periodicDims=[2])
targetSet = ShapeRectangle(g, [1.0, 2.0, -1000], [1.5, 2.5, 1000])
my_car = DubinsCar(uMode="min")
# # Test the TTR solver with obstacles
# obs = ShapeRectangle(g, [-1.0, 0.0, -1000], [0.0, 2.0, 1000])
obs = ShapeRectangle(g, [-0.2, 0.0, -1000], [0.0, 1.0, 1000])
# obs_goal = np.array([-0.5, 0.5])
# obs = CylinderShape(g, [2], np.array(obs_goal), 0.5)
po = PlotOptions(do_plot=False, plot_type="value", plotDims=[0,1], slicesCut=[25])
epsilon = 0.001
V_0_Dev = TTRSolver(my_car, g, [targetSet, obs], epsilon, po)
selected_slice = 25
V_0_Dev = np.clip(V_0_Dev, -1, 10)
# Plotting
V_slice = V_0_Dev[:, :, selected_slice]
# plot_contour(V_slice, selected_slice, -3.0, 3.0, -1.0, 4.0, 50, 50)
# Traj test
start_state = (1.0, 0.0, 0.0)
ctrl_freq = 20
traj = compute_opt_traj_TTR(my_car, g, V_0_Dev, start_state, targetSet, ctrl_freq, [2])
print(f"The planned trajectory is {traj}.")
plot_contour(V_slice, selected_slice, -3.0, 3.0, -1.0, 4.0, 50, 50, traj)



# # # TTR Example 2 from the robut_utils
# my_car = DubinsCar(uMode="min")
# g = Grid(minBounds=np.array([-6.5, -13, 0.0]), maxBounds=np.array([21.5, -1.5, 2*math.pi]),
#          dims=3, pts_each_dim=np.array([561, 231, 36]), periodicDims=[2])
# goal = np.array([-2.9, -4.6])
# targetSet = CylinderShape(g, [2], np.array(goal), 0.25)
# # obs = ShapeRectangle(g, [-2.0, 0.0, -1000], [0.0, 1.0, 1000])
# po = PlotOptions(do_plot=True, plot_type="value", plotDims=[0,1], slicesCut=[25])
# epsilon = 0.001
# V_0_Dev = TTRSolver(my_car, g, targetSet, epsilon, po)
# # V_0_Dev = TTRSolver_Dev(my_car, g, [targetSet, obs], epsilon, po)
# selected_slice = 18
# V_0_Dev = np.clip(V_0_Dev, -1, 40)
# V_slice = V_0_Dev[:, :, selected_slice]
# # Traj test
# start_state = (1.0, -2.0, 0.0)
# ctrl_freq = 20
# traj, opt_u = compute_opt_traj_TTR(my_car, g, V_0_Dev, start_state, targetSet, ctrl_freq, [2])
# # print(f"The planned trajectory is {traj}.")
# plot_contour(V_slice, selected_slice, -6.5, 21.5, -13, -1.5, 561, 231, traj)

# # TTR Example 3: 2D SIG works
# g = Grid(minBounds=np.array([-3.0, -1.0]), maxBounds=np.array([3.0, 4.0]),
#          dims=2, pts_each_dim=np.array([50, 50]), periodicDims=[])
# my_sig = Plane2D(uMode="min")
# targetSet = ShapeRectangle(g, [1.0, 2.0], [2.0, 3.0])
# obs = ShapeRectangle(g, [-2.0, 0.0], [0.0, 1.0])
# po = PlotOptions( plot_type="value", plotDims=[0,1], slicesCut=[])
# epsilon = 0.001
# V_0_Dev = TTRSolver(my_sig, g, [targetSet, obs], epsilon, po)