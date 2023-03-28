import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *
# Specify the  file that includes dynamic systems, AttackerDefender4D
from MRAG.AttackerDefender1v1 import AttackerDefender1v1 
# Plot options
from odp.Plots import PlotOptions
from odp.Plots.plotting_utilities import plot_2d, plot_isosurface
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import math
import time

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""

##################################################### EXAMPLE 4 1v1AttackerDefender ####################################
#
grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([30, 30, 30, 30])) # original 45

# Define my object dynamics
agents_1v1 = AttackerDefender1v1(uMode="min", dMode="max")  # 1v1 (4 dims dynamics)

# Avoid set, no constraint means inf
obs1_attack = ShapeRectangle(grids, [-0.1, -1.0, -1000, -1000], [0.1, -0.3, 1000, 1000])  # attacker stuck in obs1
obs2_attack = ShapeRectangle(grids, [-0.1, 0.30, -1000, -1000], [0.1, 0.60, 1000, 1000])  # attacker stuck in obs2
obs3_capture = agents_1v1.capture_set(grids, 0.1, "capture")  # attacker being captured by defender, try different radius
avoid_set = np.minimum(obs3_capture, np.minimum(obs1_attack, obs2_attack)) # original
# debugging
# avoid_set = np.minimum(obs3_capture, obs2_attack) # debug1
# obs_circle = CylinderShape(g, [2, 3], np.array([0.0, 0.5, 0.0, 0.0]), 0.3) # debug2 + debug4
# avoid_set = np.minimum(obs3_capture, obs_circle) # debug2 + debug5
# avoid_set = obs2_attack  # debug3
# avoid_set = obs_circle # debug6

# Reach set, run and see what it is!
goal1_destination = ShapeRectangle(grids, [0.6, 0.1, -1000, -1000], [0.8, 0.3, 1000, 1000])  # attacker arrives target
goal2_escape = agents_1v1.capture_set(grids, 0.1, "escape")  # attacker escape from defender
obs1_defend = ShapeRectangle(grids, [-1000, -1000, -0.1, -1000], [1000, 1000, 0.1, -0.3])  # defender stuck in obs1
obs2_defend = ShapeRectangle(grids, [-1000, -1000, -0.1, 0.30], [1000, 1000, 0.1, 0.60])  # defender stuck in obs2
reach_set = np.minimum(np.maximum(goal1_destination, goal2_escape), np.minimum(obs1_defend, obs2_defend)) # original


# Look-back length and time step
lookback_length = 4.5  # the same as 2014Mo
# t_step = 0.025
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
result = HJSolver(agents_1v1, grids, [reach_set, avoid_set], tau, compMethods, po, saveAllTimeSteps=True) # original one
# result = HJSolver(my_2agents, g, avoid_set, tau, compMethods, po, saveAllTimeSteps=True)

# We just have to project this value function to 6D case

# case1: calculate all time slices
# for i in range(len(tau)):
#     attacker1_wins_2v1 = np.zeros((30, 30, 30, 30, 30 ,30)) + np.expand_dims(result[..., i], axis=(2,3))
#     print("array type {}".format(attacker1_wins_2v1.dtype))
#     attacker1_wins_2v1 = np.array(attacker1_wins_2v1, dtype='float32')

#     attacker2_wins_2v1 = np.zeros((30, 30, 30, 30, 30 ,30)) + np.expand_dims(result[..., i], axis=(0,1))
#     attacker2_wins_2v1 = np.array(attacker2_wins_2v1, dtype='float32')
#     at_least_one_win_2v1 = np.minimum(attacker2_wins_2v1, attacker1_wins_2v1)
#     print("Saving time step {}".format(i))
#     np.save('/localhome/hha160/optimized_dp/MRAG/2v1AttackDefend_new_step{}.npy'.format(i), at_least_one_win_2v1)

# case2: calculate only the final time slice
attacker1_wins_2v1 = np.zeros((30, 30, 30, 30, 30, 30)) + np.expand_dims(result[..., 0], axis = (2, 3))
attacker1_wins_2v1 = np.array(attacker1_wins_2v1, dtype='float32')
attacker2_wins_2v1 = np.zeros((30, 30, 30, 30, 30, 30)) + np.expand_dims(result[..., 0], axis = (0, 1))
at_least_one_win_2v1 = np.minimum(attacker1_wins_2v1, attacker2_wins_2v1)
print(f"The shape of the at_least_one_win_2v1 is {at_least_one_win_2v1.shape}. \n ")
# np.save('/localhome/hha160/optimized_dp/MRAG/2v1AttackDefend.npy', at_least_one_win_2v1)
np.save('2v1AttackDefend.npy', at_least_one_win_2v1)