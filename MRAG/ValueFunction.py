import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems, AttackerDefender4D
from odp.dynamics.AttackerDefender4D import AttackerDefender4D 
# Plot options
from odp.Plots import PlotOptions
from odp.Plots.plotting_utilities import plot_2d, plot_isosurface
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

g = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45])) # original 45,; 80 doesn't work

# Define my object dynamics
my_2agents = AttackerDefender4D(uMode="min", dMode="max")  # todo the dynamics may have some bugs

# Avoid set, no constraint means inf
obs1_attack = ShapeRectangle(g, [-0.1, -1.0, -1000, -1000], [0.1, -0.3, 1000, 1000])  # attacker stuck in obs1
obs2_attack = ShapeRectangle(g, [-0.1, 0.30, -1000, -1000], [0.1, 0.60, 1000, 1000])  # attacker stuck in obs2
obs3_capture = my_2agents.capture_set(g, 0.05, "capture")  # attacker being captured by defender, try different radius
avoid_set = np.minimum(obs3_capture, np.minimum(obs1_attack, obs2_attack)) # original
# debugging
# avoid_set = np.minimum(obs3_capture, obs2_attack) # debug1
# obs_circle = CylinderShape(g, [2, 3], np.array([0.0, 0.5, 0.0, 0.0]), 0.3) # debug2 + debug4 
# avoid_set = np.minimum(obs3_capture, obs_circle) # debug2 + debug5
# avoid_set = obs2_attack  # debug3
# avoid_set = obs_circle # debug6

# Reach set, run and see what it is!
goal1_destination = ShapeRectangle(g, [0.6, 0.1, -1000, -1000], [0.8, 0.3, 1000, 1000])  # attacker arrives target
goal2_escape = my_2agents.capture_set(g, 0.05, "escape")  # attacker escape from defender
obs1_defend = ShapeRectangle(g, [-1000, -1000, -0.1, -1000], [1000, 1000, 0.1, -0.3])  # defender stuck in obs1
obs2_defend = ShapeRectangle(g, [-1000, -1000, -0.1, 0.30], [1000, 1000, 0.1, 0.60])  # defender stuck in obs2
reach_set = np.minimum(np.maximum(goal1_destination, goal2_escape), np.minimum(obs1_defend, obs2_defend)) # original
# debugging
# reach_set = np.minimum(np.maximum(goal1_destination, goal2_escape), obs2_defend) # debug1
# obs_circle_defend = CylinderShape(g, [0, 1], np.array([0.0, 0.5, 0.0, 0.0]), 0.3) # debug2 + debug4
# goal1_circle = CylinderShape(g, [2, 3], np.array([0.5, 0.5, 0.0, 0.0]), 0.2) # debug5
# reach_set = np.minimum(goal1_destination, obs2_defend)  # debug3
# reach_set = np.minimum(goal1_destination, obs_circle_defend) # debug4
# reach_set = np.minimum(np.maximum(goal1_circle, goal2_escape), obs_circle_defend) # debug5
# reach_set = np.minimum(goal1_circle, obs_circle_defend) # debug6
# reach_set = np.minimum(goal1_destination, np.minimum(obs1_defend, obs2_defend))  
# reach_set = np.minimum(goal1_destination, obs_circle_defend) 
# reach_set = goal1_destination 

# Look-back length and time step
lookback_length = 1.0  # try 1.5, 2.0, 2.5, 3.0, 5.0, 6.0, 8.0
t_step = 0.05

# Actual calculation process, needs to add new plot function to draw a 2D figure
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# while plotting make sure the len(slicesCut) + len(plotDims) = grid.dims
po = PlotOptions(do_plot=False, plot_type="2d_plot", plotDims=[0, 1], slicesCut=[22, 22])
# plot the 2 obs
# plot_isosurface(g, np.minimum(obs1_attack, obs2_attack), po)
# plot_isosurface(g, obs3_capture, po)

# In this example, we compute a Reach-Avoid Tube
compMethods = {"TargetSetMode": "minVWithVTarget", "ObstacleSetMode": "maxVWithObstacle"} # original one
# compMethods = {"TargetSetMode": "minVWithVTarget"}
result = HJSolver(my_2agents, g, [reach_set, avoid_set], tau, compMethods, po, saveAllTimeSteps=True) # original one
# result = HJSolver(my_2agents, g, avoid_set, tau, compMethods, po, saveAllTimeSteps=True)

print(f'The shape of the value function is {result.shape} \n')
# save the value function
np.save('1v1AttackDefend.npy', result)


# # Compute spatial derivatives at every state
# last_time_step_result = result[..., 0]
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
