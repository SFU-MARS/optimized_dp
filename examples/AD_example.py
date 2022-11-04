# import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCapture
from odp.dynamics import DubinsCar4D2
from odp.dynamics.AttackerDefender4D import AttackerDefender4D
# Plot options
from odp.Plots import PlotOptions
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray

import math

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""

##################################################### EXAMPLE 4 1v1AttackerDefender #####################################################

g = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))

# Define my object dynamics
my_2agents = AttackerDefender4D(uMode="min", dMode="max")

# Reach set, how to define the goal state to the defender or in 4 dimensions?
goal = ShapeRectangle(g, [0.6, 0.1, -1.0, -1.0], [0.8, 0.3, 1.0, 1.0])

# Avoid set, not finished yet
obs1 = ShapeRectangle(g, [-0.1, -1.0, -0.1, -1.0], [0.1, -0.3, 0.1, -0.3])
obs2 = ShapeRectangle(g, [-0.2, 0.25, -2000, -2000], [0.1, -0.3, 0.1, -0.3])
obstacle = np.min(obs1, obs2)

# Look-back length and time step
lookback_length = 1.5  # try 2.0 the output figure is none
t_step = 0.05

# Actual calculation process, needs to add new plot function to draw a 2D figure
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

po = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2],
                 slicesCut=[29])

# In this example, we compute a Reach-Avoid Tube
compMethods = {"TargetSetMode": "minVWithVTarget",
               "ObstacleSetMode": "maxVWithObstacle"}


result = HJSolver(my_2agents, g, [goal, obstacle], tau, compMethods, po, saveAllTimeSteps=True)
# print(f'The shape of the result is {result.shape}')

# last_time_step_result = result[..., 0]
#
# # Compute spatial derivatives at every state
# x_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=1, accuracy="low")
# y_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=2, accuracy="low")
# v_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=3, accuracy="low")
# T_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=4, accuracy="low")
#
# # Let's compute optimal control at some random idices
# spat_deriv_vector = (x_derivative[10,20,15,15], y_derivative[10,20,15,15],
#                      v_derivative[10,20,15,15], T_derivative[10,20,15,15])
#
# # Compute the optimal control
# opt_a, opt_w = my_2agents.optCtrl_inPython(spat_deriv_vector)
# print("Optimal accel is {}\n".format(opt_a))
# print("Optimal rotation is {}\n".format(opt_w))


