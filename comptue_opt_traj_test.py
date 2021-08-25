from dynamics.compute_opt_traj import compute_opt_traj
import numpy as np
# Utility functions to initialize the problem
from Grid.GridProcessing import Grid
from Shapes.ShapesFunctions import *
from plot_options import *
from Plots.plotting_utilities import *
from dynamics.DubinsCar import DubinsCar
from solver import HJSolver

import math

# dCar = DubinsCar([0, 0, 0], wMax=1, speed=1)
""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""

# Scenario 1
g = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([40, 40, 40]), [2])
dyn_sys = DubinsCar()
Initial_value_f = CylinderShape(g, [2], np.zeros(3), 1)

# Look-back lenght and time step
lookback_length = 2.0
t_step = 0.05
small_number = 1e-5
# this is just need to include the endpoint, np.linspace is a fine alternative
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

"""
Assign one of the following strings to `compMethod` to specify the characteristics of computation
"none" -> compute Backward Reachable Set
"minVWithV0" -> compute Backward Reachable Tube
"maxVWithVInit" -> compute max V over time
"minVWithVInit" compute min V over time
"""

# po2 = PlotOptions("3d_plot", [0, 1, 2], [])
# V_all_t = HJSolver(dyn_sys, g, Initial_value_f, tau, "minVWithV0", po2, save_all_t=True)
# np.save("V_all_t_minVWithV0", V_all_t)


# from test_plotter import plot
# plot(g, V_all_t[..., 0], dims=[0, 1, 2])
# # plot(g, V_all_t[..., 1], dims=[0, 1, 2])
# plot(g, V_all_t[..., -1], dims=[0, 1, 2])

def main():
    V_all_t = np.load("V_all_t_minVWithV0.npy")
    print(V_all_t.shape)
    from test_plotter import plot
    plot(g, V_all_t[..., 0], dims=[0, 1, 2])
    # plot(g, V_all_t[..., 1], dims=[0, 1, 2])
    plot(g, V_all_t[..., -1], dims=[0, 1, 2])
    # test_dyn_sys = DubinsCar(x=np.array([2, 2, -np.pi])
    # theta_idx = np.abs(g.grid_points[2] - (-np.pi)).argmin()
    #
    # compute_opt_traj(g, V_all_t, tau, dyn_sys)

if __name__ in "__main__":
    main()
