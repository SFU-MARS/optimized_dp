import time
import os
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar4D
# Plot options
from odp.Plots import PlotOptions, plot_isosurface, plot_valuefunction
from odp.compute_trajectory import spa_deriv
from utils import *

# Solver core
from odp.solver import HJSolver, TTRSolver
import math


# # Define grid
grid = Grid(np.array([-3.0, -1.0, 0.0, -math.pi]), np.array([3.0, 4.0, 4.0, math.pi]), 4, np.array([60, 60, 20, 36]), [3])

# Define my object
my_car = DubinsCar4D(x=[0,0,0,0], uMin = [-1,-1], uMax = [1,1], dMin = [0.0, 0.0], dMax=[0.0, 0.0], uMode="min", dMode="max")

# Use the grid to initialize initial value function
goal_area = CylinderShape(grid, [2,3], np.array([2., 2., 0., 0.]), 0.8)
obs_area = ShapeRectangle(grid, np.array([-1, -1, -1.0, -10]), np.array([1, 1, 5.0, 10]))

# HJ solver setting
tol = 0.001

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

po = PlotOptions(do_plot=True, plot_type="value", plotDims=[0,1],
                  slicesCut=[10, 18], save_fig=True, filename=f"{current_directory}/dubins4d_ttr.png")

ttr_value = TTRSolver(my_car, grid, [goal_area, obs_area], tol, po)

np.save(f"{current_directory}/hj_values/dubins4d_ttr.npy", ttr_value)
