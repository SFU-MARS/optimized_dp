import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *
# Specify the  file that includes dynamic systems
# from MRAG.AttackerDefender2v1 import AttackerDefender2v1
from AttackerDefender2v1 import *
# Plot options
from odp.Plots import PlotOptions
from odp.Plots.plotting_utilities import plot_2d, plot_isosurface
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import math
import gc
import os, psutil


""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""

##################################################### EXAMPLE 5 2v1AttackerDefender ####################################

grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30])) # original 45, on mars-14 20 is the upper bound
process = psutil.Process(os.getpid())
print("1. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

# Define my object dynamics
agents_2v1 = AttackerDefender2v1(uMode="min", dMode="max")  # 2v1 (6 dim dynamics)
# Avoid set, no constraint means inf
obs1_a1 = ShapeRectangle(grids, [-0.1, -1.0, -1000, -1000, -1000, -1000], [0.1, -0.3, 1000, 1000, 1000, 1000])  # a1 get stuck in the obs1

process = psutil.Process(os.getpid())
print("2. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

obs2_a1 = ShapeRectangle(grids, [-0.1, 0.30, -1000, -1000, -1000, -1000], [0.1, 0.60, 1000, 1000, 1000, 1000])  # a1 get stuck in the obs2
process = psutil.Process(os.getpid())
print("3. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

tmp2 = np.minimum(obs1_a1, obs2_a1)
del obs1_a1
del obs2_a1
gc.collect()

obs1_a2 = ShapeRectangle(grids, [-1000, -1000, -0.1, -1.0, -1000, -1000], [1000, 1000, 0.1, -0.3, 1000, 1000])  # a2 get stuck in the obs1
process = psutil.Process(os.getpid())
print("4. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

obs2_a2 = ShapeRectangle(grids, [-1000, -1000, -0.1, 0.30, -1000, -1000], [1000, 1000, 0.1, 0.60, 1000, 1000])  # a2 get stuck in the obs2
process = psutil.Process(os.getpid())
print("5. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

tmp1 = np.minimum(obs1_a2, obs2_a2)
del obs1_a2
del obs2_a2
gc.collect()

process = psutil.Process(os.getpid())
print("6. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

tmp3 = np.minimum(tmp1, tmp2)
del tmp1
del tmp2
gc.collect()

process = psutil.Process(os.getpid())
print("7. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

capture_a1 = agents_2v1.capture_set1(grids, 0.1, "capture")  # a1 is captured
process = psutil.Process(os.getpid())
print("8. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

capture_a2 = agents_2v1.capture_set2(grids, 0.1, "capture")  # a2 is captured
process = psutil.Process(os.getpid())
print("10. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes
tmp4 = np.minimum(capture_a1, capture_a2)
del capture_a1
del capture_a2
gc.collect()

avoid_set = np.minimum(tmp3, tmp4) # is order necessary?
process = psutil.Process(os.getpid())
print("11. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes
# Save memory by getting rid of the intial array

del tmp3
del tmp4
gc.collect()

process = psutil.Process(os.getpid())
print("Hello Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes



# Reach set, run and see what it is!
goal1_destination = ShapeRectangle(grids, [0.6, 0.1, 0.6, 0.1, -1000, -1000],
                                   [0.8, 0.3, 0.8, 0.3, 1000, 1000])  # a1 and a2 both arrive the goal
# np.save('goal1_destination.npy', goal1_destination)

# goal1_destination = ShapeRectangle(grids, [0.6, 0.1, -1000, -1000, -1000, -1000],
#                                    [0.8, 0.3, 1000, 1000, 1000, 1000])  # a1 and a2 both arrive the goal
# np.save('goal1_destination.npy', goal1_destination)

escape_a1 = agents_2v1.capture_set1(grids, 0.1, "escape")  # a1 escape
escape_a2 = agents_2v1.capture_set2(grids, 0.1, "escape")  # a2 escape

obs1_defend = ShapeRectangle(grids, [-1000, -1000, -1000, -1000, -0.1, -1.0],
                             [1000, 1000, 1000, 1000, 0.1, -0.3])  # defender stuck in obs1
obs2_defend = ShapeRectangle(grids, [-1000, -1000, -1000, -1000, -0.1, 0.30],
                             [1000, 1000, 1000, 1000, 0.1, 0.60])  # defender stuck in obs2

process = psutil.Process(os.getpid())
print("12. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

another_tmp1 = np.maximum(goal1_destination, escape_a1)
another_tmp2 = np.maximum(goal1_destination, escape_a2)
del escape_a1
del escape_a2
del goal1_destination
gc.collect()


another_tmp3 = np.maximum(another_tmp1, another_tmp2)
del another_tmp2
del another_tmp1
gc.collect()

another_tmp4 = np.minimum(obs1_defend, obs2_defend)
del obs2_defend
del obs1_defend
gc.collect()

reach_set = np.minimum(another_tmp3, another_tmp4)
del another_tmp3
del another_tmp4
gc.collect()

process = psutil.Process(os.getpid())
print("13. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

# Look-back length and time step
lookback_length = 1.5  # try 1.5, 2.0, 2.5, 3.0, 5.0, 6.0, 8.0
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
np.save('/localhome/hha160/optimized_dp/MRAG/2v1AttackDefend.npy', result)
