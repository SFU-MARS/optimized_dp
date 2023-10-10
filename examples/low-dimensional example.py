import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import Plane2D, Plane1D
# Plot options
from odp.Plots import PlotOptions
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray

import math

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- System dynamics for computation
- Initialize plotting option
- Call HJSolver function
"""

##################################################### 2D EXAMPLE #####################################################
# STEP 1: Define grid
grid_min = np.array([-4.0, -4.0])
grid_max = np.array([4.0, 4.0])
dims = 2
N = np.array([40, 40])
g = Grid(grid_min, grid_max, dims, N)

# STEP 2: Generate initial values for grid using shape functions
target_min = np.array([-1.0, -1.0])
target_max = np.array([1.0, 1.0])
Initial_value_f = ShapeRectangle(g, target_min, target_max)

# STEP 3: Time length for computations
Lookback_length = 2.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=Lookback_length + small_number, step=t_step)

# STEP 4: System dynamics for computation
sys = Plane2D()

# STEP 5: Initialize plotting option
po2 = PlotOptions(do_plot=True, plot_type="2d_plot", plotDims=[0,1],
                  slicesCut=[], colorscale="Hot")

# STEP 6: Call HJSolver function
compMethod = { "TargetSetMode": "None"}
result_2 = HJSolver(sys, g, Initial_value_f, tau, compMethod, po2)



# ##################################################### 1D EXAMPLE #####################################################
# # STEP 1: Define grid
# grid_min_1 = np.array([-4.0])
# grid_max_1 = np.array([4.0])
# dims_1 = 1
# N_1 = np.array([40])
# g_1 = Grid(grid_min_1, grid_max_1, dims_1, N_1)

# # STEP 2: Generate initial values for grid using shape functions
# target_min_1 = np.array([-1.0])
# target_max_1 = np.array([1.0])
# Initial_value_f_1 = ShapeRectangle(g_1, target_min_1, target_max_1)

# # STEP 3: Time length for computations
# Lookback_length = 2.0
# t_step = 0.05

# small_number = 1e-5
# tau = np.arange(start=0, stop=Lookback_length + small_number, step=t_step)

# # STEP 4: System dynamics for computation
# sys_1 = Plane1D()

# # STEP 5: Initialize plotting option
# po2 = PlotOptions(do_plot=True, plot_type="1d_plot", plotDims=[0],
#                   slicesCut=[])

# # STEP 6: Call HJSolver function
# compMethod = { "TargetSetMode": "None"}

# print(1)
# result_1 = HJSolver(sys_1, g_1, Initial_value_f_1, tau, compMethod, po2)

