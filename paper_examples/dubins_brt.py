import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCapture, Plane2D, Plane1D, DubinsCar4D, DubinsCar
# Plot options
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray

import math
import os

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- System dynamics for computation
- Initialize plotting option
- Call HJSolver function

Note: If run on the server, please save the result and use the plot function on your local machine
"""

if os.path.exists("plots") == False:
    os.mkdir("plots")
        
# ##################################################### 4D EXAMPLE #####################################################
# # STEP 1: Define grid
# grid_min = np.array([-4.0, -4.0, -4.0, -math.pi])
# grid_max = np.array([4.0, 4.0, 4.0, math.pi])
# dims = 4
# N = np.array([80, 80, 80, 80])
# pd=[3]
# g = Grid(grid_min, grid_max, dims, N, pd)

# # STEP 2: Generate initial values for grid using shape functions
# center = np.zeros(dims)
# radius = 0.5
# ignore_dims = [3]
# Initial_value_f = CylinderShape(g, ignore_dims, center, radius)

# # STEP 3: Time length for computations
# Lookback_length = 1.0
# t_step = 0.05

# small_number = 1e-5
# tau = np.arange(start=0, stop=Lookback_length + small_number, step=t_step)

# # STEP 4: System dynamics for computation
# sys4D = DubinsCar4D(uMode="max", dMode="min")  

# # STEP 5: Initialize plotting option
# po = PlotOptions(do_plot=True, plot_type="set", plotDims=[0,1], slicesCut=[45, 79], colorscale="Bluered", save_fig=False, filename="plots/4D_0_sublevel_set", interactive_html=True)

# # STEP 6: Call HJSolver function
# compMethod = { "TargetSetMode": "None"}
# result_3 = HJSolver(sys4D, g, Initial_value_f, tau, compMethod, po, saveAllTimeSteps=True)


# ##################################################### 3D EXAMPLE #####################################################
# # STEP 1: Define grid
grid_min = np.array([-4.0, -4.0, -math.pi])
grid_max = np.array([4.0, 4.0, math.pi])
dims = 3
N = np.array([150, 150, 150])
pd=[2]
g = Grid(grid_min, grid_max, dims, N, pd)

# STEP 2: Generate initial values for grid using shape functions
center = np.zeros(dims)
radius = 1.0
ignore_dims = [2]
Initial_value_f = CylinderShape(g, ignore_dims, center, radius)

# STEP 3: Time length for computations
Lookback_length = 1.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=Lookback_length + small_number, step=t_step)

# STEP 4: System dynamics for computation
sys = DubinsCar(uMode="max", dMode="min", wMax=2.0)

# STEP 5: Initialize plotting option
po1 = PlotOptions(do_plot=True, plot_type="set", plotDims=[0,1], slicesCut=[149])

# STEP 6: Call HJSolver function
compMethod = { "TargetSetMode": "maxVWithObstacle"}
result_3 = HJSolver(sys, g, Initial_value_f, tau, compMethod, po1, saveAllTimeSteps=True)

# '''
# Test downsample function
# '''
# # print(result_3[:,:,:,1].shape)
# # print(g.dims)
# # g_out, data_out = downsample(g, result_3, [2,2,2])

# # print(data_out.shape)


# # # While file needs to be saved locally, set save_fig=True and filename, recommend to set interactive_html=True for better visualization
# # po2 = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1,2],
# #                   slicesCut=[1], colorscale="Bluered", save_fig=True, filename="plots/3D_0_sublevel_set", interactive_html=True)

# # # STEP 6: Call Plotting function
# # plot_isosurface(g, result_3, po2)