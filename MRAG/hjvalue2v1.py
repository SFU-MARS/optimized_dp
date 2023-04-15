import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *
# Specify the  file that includes dynamic systems, AttackerDefender4D
from MRAG.AttackerDefender1v1 import AttackerDefender1v1
from AttackerDefender2v1 import *
# Plot options
from odp.Plots import PlotOptions
from odp.Plots.plotting_utilities import plot_2d, plot_isosurface
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import math
import time
import gc
import os, psutil


""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""

##################################################### EXAMPLE 4 2v1AttackerDefender ####################################
grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
             6, np.array([30, 30, 30, 30, 30, 30]))
process = psutil.Process(os.getpid())
print("1. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

# First load the 4D reach-avoid set
RA_1V1 = np.load("1v1AttackDefend_g30_speed1.npy")

# Define my object dynamics
agents_2v1 = AttackerDefender2v1(uMode="min", dMode="max")  # 2v1 (6 dim dynamics)
# Avoid set, no constraint means inf
obs1_a1 = ShapeRectangle(grids, [-0.1, -1.0, -1000, -1000, -1000, -1000], [0.1, -0.3, 1000, 1000, 1000, 1000])  # a1 get stuck in the obs1
obs1_a1 = np.array(obs1_a1, dtype='float32')
process = psutil.Process(os.getpid())
print("2. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

obs2_a1 = ShapeRectangle(grids, [-0.1, 0.30, -1000, -1000, -1000, -1000], [0.1, 0.60, 1000, 1000, 1000, 1000])  # a1 get stuck in the obs2
obs2_a1 = np.array(obs2_a1, dtype='float32')
process = psutil.Process(os.getpid())
print("3. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

obs_a1 = np.minimum(obs1_a1, obs2_a1)
obs_a1 = np.array(obs_a1, dtype='float32')
del obs1_a1
del obs2_a1
gc.collect()

capture_a1 = agents_2v1.capture_set1(grids, 0.1, "capture")  # a1 is captured
capture_a1 = np.array(capture_a1, dtype='float32')

a1_captured = np.minimum(capture_a1, obs_a1)
a1_captured = np.array(a1_captured, dtype='float32')

# TODO: check the axis is right
# Backproject 4D reach-avoid array to 6D
# The losing conditions is complement of winning conditions of attacker 2
a2_lose_after_a1 = -(np.zeros((30, 30, 30, 30, 30, 30)) + np.expand_dims(RA_1V1, axis = (0, 1)))
a2_lose_after_a1 = np.array(a2_lose_after_a1, dtype='float32')
process = psutil.Process(os.getpid())
print("4. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

a1_captured_then_a2_lose = np.maximum(a1_captured, a2_lose_after_a1)
a1_captured_then_a2_lose = np.array(a1_captured_then_a2_lose, dtype='float32')
del a1_captured
del a2_lose_after_a1

# Attacker 2 avoid set
obs1_a2 = ShapeRectangle(grids, [-1000, -1000, -0.1, -1.0, -1000, -1000], [1000, 1000, 0.1, -0.3, 1000, 1000])  # a2 get stuck in the obs1
process = psutil.Process(os.getpid())
print("5. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

obs2_a2 = ShapeRectangle(grids, [-1000, -1000, -0.1, 0.30, -1000, -1000], [1000, 1000, 0.1, 0.60, 1000, 1000])  # a2 get stuck in the obs2
process = psutil.Process(os.getpid())
print("6. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

obs_a2 = np.minimum(obs1_a2, obs2_a2)
obs_a2 = np.array(obs_a2, dtype='float32')
del obs1_a2
del obs2_a2
gc.collect()
process = psutil.Process(os.getpid())
print("7. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

capture_a2 = agents_2v1.capture_set2(grids, 0.1, "capture")  # a2 is captured
# Convert to float 32
capture_a2 = np.array(capture_a2, dtype='float32')
process = psutil.Process(os.getpid())

a2_captured = np.minimum(obs_a2, capture_a2)
a2_captured = np.array(a2_captured, dtype='float32')
del obs_a2
del capture_a2

# The losing conditions is complement of winning conditions of attacker 1
a1_lose_after_a2 = -(np.zeros((30, 30, 30, 30, 30, 30)) + np.expand_dims(RA_1V1, axis = (2, 3)))
a1_lose_after_a2 = np.array(a1_lose_after_a2, dtype='float32')
process = psutil.Process(os.getpid())
print("8. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

a2_captured_then_a1_lose = np.maximum(a1_lose_after_a2, a2_captured)
a2_captured_then_a1_lose = np.array(a2_captured_then_a1_lose, dtype='float32')
del a1_lose_after_a2
del a2_captured

avoid_set = np.minimum(a2_captured_then_a1_lose, a1_captured_then_a2_lose)
np.array(avoid_set, dtype='float32')
del a2_captured_then_a1_lose
del a1_captured_then_a2_lose

print("9. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

# Reach set, at least one of them manage to reach the target
# goal1_destination = ShapeRectangle(grids, [0.6, 0.1, 0.6, 0.1, -1000, -1000],
#                                    [0.8, 0.3, 0.8, 0.3, 1000, 1000])  # a1 and a2 both arrive the goal
# np.save('goal1_destination.npy', goal1_destination)

# goal1_destination = ShapeRectangle(grids, [0.6, 0.1, -1000, -1000, -1000, -1000],
#                                    [0.8, 0.3, 1000, 1000, 1000, 1000])  # a1 and a2 both arrive the goal
# np.save('goal1_destination.npy', goal1_destination)

a1_goal = ShapeRectangle(grids, [0.6, 0.1, -1000, -1000, -1000, -1000],
                                [0.8, 0.3,  1000,  1000,  1000,  1000])  # a1 is at goal
a1_is_safe = agents_2v1.capture_set1(grids, 0.1, "escape")  # a1 is 0.1 away from defender
a1_wins = np.maximum(a1_goal, a1_is_safe)
a1_wins = np.array(a1_wins, dtype='float32')

a2_goal = ShapeRectangle(grids, [-1000, -1000, 0.6, 0.1, -1000, -1000],
                                [1000,   1000, 0.8, 0.3 , 1000,  1000])  # a2 is at goal
a2_is_safe = agents_2v1.capture_set2(grids, 0.1, "escape")  # a2 escape
a2_wins = np.maximum(a2_goal, a2_is_safe)
a2_wins = np.array(a2_wins, dtype='float32')

a1_or_a2_wins = np.minimum(a1_wins, a2_wins)
del a2_wins
del a1_wins
a1_or_a2_wins = np.array(a1_or_a2_wins, dtype='float32')
gc.collect()

obs1_defend = ShapeRectangle(grids, [-1000, -1000, -1000, -1000, -0.1, -1.0],
                                    [1000, 1000, 1000, 1000, 0.1, -0.3])  # defender stuck in obs1
obs2_defend = ShapeRectangle(grids, [-1000, -1000, -1000, -1000, -0.1, 0.30],
                                    [1000, 1000, 1000, 1000, 0.1, 0.60])  # defender stuck in obs2

d_lose = np.minimum(obs1_defend, obs2_defend)
d_lose = np.array(d_lose, dtype='float32')
del obs2_defend
del obs1_defend
gc.collect()
process = psutil.Process(os.getpid())
print("10. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

reach_set = np.minimum(d_lose, a1_or_a2_wins)
reach_set = np.array(reach_set, dtype='float32')
del d_lose
del a1_or_a2_wins
gc.collect()
process = psutil.Process(os.getpid())
print("11. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes


# Look-back length and time step
lookback_length = 4.5  # try 1.5, 2.0, 2.5, 3.0, 5.0, 6.0, 8.0
t_step = 0.025

# Actual calculation process, needs to add new plot function to draw a 2D figure
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# while plotting make sure the len(slicesCut) + len(plotDims) = grid.dims
po = PlotOptions(do_plot=False, plot_type="2d_plot", plotDims=[0, 1], slicesCut=[22, 22])

# In this example, we compute a Reach-Avoid Tube
compMethods = {"TargetSetMode": "minVWithVTarget", "ObstacleSetMode": "maxVWithObstacle"} # original one
result = HJSolver(agents_2v1, grids, [reach_set, avoid_set], tau, compMethods, po, saveAllTimeSteps=False) # original one

print(f'The shape of the value function is {result.shape} \n')
# save the value function

# np.save('2v1AttackDefend.npy', result)
print("The calculation is done! \n")

np.save('2v1AttackDefend_speed15.npy', result)