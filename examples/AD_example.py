# import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
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

##################################################### EXAMPLE 4 1v1AttackerDefender ####################################

g = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([30, 30, 30, 30]))

# Define my object dynamics
my_2agents = AttackerDefender4D(uMode="min", dMode="max")

# Avoid set, not finished yet
obs1_attack = ShapeRectangle(g, [-0.1, -1.0, -100, -100], [0.1, -0.3, 100, 100])  # attacker stuck in obs1
obs2_attack = ShapeRectangle(g, [-0.2, 0.25, -100, -100], [0.1, -0.3, 100, 100])  # attacker stuck in obs2
obs3_capture = my_2agents.capture_set(g, 0.1, "capture")  # attacker being captured by defender
avoid_set = np.minimum(obs3_capture, np.minimum(obs1_attack, obs2_attack))

# Reach set, run and see what it is!
goal1_destination = ShapeRectangle(g, [0.6, 0.1, -1.0, -1.0], [0.8, 0.3, 1.0, 1.0])  # attacker arrives target region
goal2_escape = my_2agents.capture_set(g, 0.1, "escape")  # attacker escape from defender
obs1_defend = ShapeRectangle(g, [-100, -100, -0.1, -1.0], [100, 2000, 0.1, -0.3])  # defender stuck in obs1
obs2_defend = ShapeRectangle(g, [-100, -100, -0.2, 0.25], [100, 2000, 0.1, -0.3])  # defender stuck in obs2
reach_set = np.minimum(np.maximum(goal1_destination, goal2_escape), np.minimum(obs1_defend, obs2_defend))


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


result = HJSolver(my_2agents, g, [reach_set, avoid_set], tau, compMethods, po, saveAllTimeSteps=True)
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


