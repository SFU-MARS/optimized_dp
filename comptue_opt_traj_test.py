from dynamics.compute_opt_traj import compute_opt_traj
import numpy as np
# Utility functions to initialize the problem
from Grid.GridProcessing import Grid
from Shapes.ShapesFunctions import *
from plot_options import *
from Plots.plotting_utilities import *
from dynamics.DubinsCar import DubinsCar
from solver import HJSolver

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import math

def main():
    grid = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([40, 40, 40]), [2])
    dyn_sys = DubinsCar()
    Initial_value_f = CylinderShape(grid, [2], np.zeros(3), 1)

    lookback_length = 2.0
    t_step = 0.05
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
    V_all_t = HJSolver(dyn_sys, grid, Initial_value_f, tau, "minVWithV0", PlotOptions("3d_plot", [0, 1, 2], []),
                       save_all_t=True)

    from test_plotter_3d import plot
    # plot(grid, V_all_t[..., -1], dims=[0,1,2])
    # np.save("V_all_t_minVWithV0", V_all_t)

    # V_all_t = np.load("V_all_t_minVWithV0.npy")

    test_dyn_sys = DubinsCar(x=np.array([1.75, 1.75, -np.pi]))
    theta_idx = np.abs(grid.grid_points[2] - (-np.pi)).argmin()
    print(grid.get_value(V_all_t[..., -1], test_dyn_sys.x))
    #     print("Inital state not in BRT/BRS")g
    #     exit()
    traj = compute_opt_traj(grid, V_all_t, tau, test_dyn_sys)
    print(traj)

    plot_trajectory(grid, V_all_t, traj)


if __name__ in "__main__":
    main()
