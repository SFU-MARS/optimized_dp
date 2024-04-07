import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *
# Specify the  file that includes dynamic systems, AttackerDefender4D
from AttackerDefender1v3_7D import *
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
- 1. Initialize the grids
- 2. Initialize the dynamics
- 3. Instruct the avoid set and reach set
- 4. Set the look-back length and time step
- 5. Call HJSolver function
- 6. Save the value function
"""

##################################################### EXAMPLE 4 2v1AttackerDefender ####################################
# Record the time of whole process
start_time = time.time()
print("The start time is {}".format(start_time))

# 1. Initialize the grids
# grid_size = 21
grid_size1 = 20
grid_size2 = 21

# grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
#              7, np.array([grid_size, grid_size, grid_size, grid_size, grid_size, grid_size, grid_size])) 

# grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
#              7, np.array([grid_size1, grid_size1, grid_size1, grid_size1, grid_size1, grid_size1, grid_size2]))  # grid = 9^4*10^4

# grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
#              7, np.array([grid_size, grid_size, grid_size, grid_size, grid_size, grid_size, 22])) 

grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
             7, np.array([grid_size1, grid_size1, grid_size1, grid_size2, grid_size2, grid_size2, grid_size2]))  # 

process = psutil.Process(os.getpid())
print("1. Gigabytes consumed by the grids is {}".format(process.memory_info().rss/(1e9)))  # in bytes

# 2. Initialize the dynamics
agents_1v3_7D = AttackerDefender1v3_7D(uMode="min", dMode="max")  # 1 vs.3 with 1 defender restricted (7 dim dynamics)

# 3. Instruct the avoid set and reach set
# 3.1 Avoid set, no constraint means inf
capture_a1d1 = agents_1v3_7D.capture_set1(grids, 0.1, "capture")  # a1 is captured by d1
capture_a1d1 = np.array(capture_a1d1, dtype='float32')

capture_a1d2 = agents_1v3_7D.capture_set2(grids, 0.1, "capture")  # a1 is captured by d2
capture_a1d2 = np.array(capture_a1d2, dtype='float32')

capture_a1d3 = agents_1v3_7D.capture_set3(grids, 0.1, "capture")  # a1 is captured by d3
capture_a1d3 = np.array(capture_a1d3, dtype='float32')

avoid_set = np.minimum(capture_a1d1, np.minimum(capture_a1d2, capture_a1d3))
avoid_set = np.array(avoid_set, dtype='float32')

del capture_a1d1
del capture_a1d2
del capture_a1d3

process = psutil.Process(os.getpid())
print("2. Gigabytes consumed of the avoid set is {}".format(process.memory_info().rss/(1e9)))  # in bytes
gc.collect()

# 3.2 Reach set
goal1_destination = ShapeRectangle(grids, [0.6, 0.1, -1000, -1000, -1000, -1000, -1000], [0.8, 0.3, 1000, 1000, 1000, 1000, 1000])  # attacker arrives target
goal2_escape_d1 = agents_1v3_7D.capture_set1(grids, 0.1, "escape")  # a1 escape from d1
goal3_escape_d2 = agents_1v3_7D.capture_set2(grids, 0.1, "escape")  # a1 escape from d2
goal4_escape_d3 = agents_1v3_7D.capture_set3(grids, 0.1, "escape")  # a1 escape from d3

reach_set = np.maximum(goal1_destination, np.maximum(goal2_escape_d1, np.maximum(goal3_escape_d2, goal4_escape_d3)))
reach_set = np.array(reach_set, dtype='float32')

process = psutil.Process(os.getpid())
print("3. Gigabytes consumed of the reach set is {}".format(process.memory_info().rss/(1e9)))  # in bytes
gc.collect()

# 4. Set the look-back length and time step
lookback_length = 1.0  # try 1.5, 2.0, 2.5, 3.0, 5.0, 6.0, 8.0  
t_step = 0.025

# Actual calculation process, needs to add new plot function to draw a 2D figure
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# while plotting make sure the len(slicesCut) + len(plotDims) = grid.dims
po = PlotOptions(do_plot=False, plot_type="2d_plot", plotDims=[0, 1], slicesCut=[22, 22])

# 5. Call HJSolver function
compMethods = {"TargetSetMode": "minVWithVTarget", "ObstacleSetMode": "maxVWithObstacle"} # original one
solve_start_time = time.time()
result = HJSolver(agents_1v3_7D, grids, [reach_set, avoid_set], tau, compMethods, po, saveAllTimeSteps=False) # original one

process = psutil.Process(os.getpid())
print(f"The CPU memory used during the calculation of the value function is {process.memory_info().rss/(1e9): .2f} GB.")  # in bytes

solve_end_time = time.time()
print(f'The shape of the value function is {result.shape} \n')
print(f"The size of the value function is {result.nbytes / (1e9): .2f} GB or {result.nbytes/(1e6)} MB.")
print(f"The time of solving HJ is {solve_end_time - solve_start_time} seconds.")
print(f'The shape of the value function is {result.shape} \n')

print("The calculation is done! \n")

# # 6. Save the value function
# np.save(f'1v3_7DAttackDefend_g{grid_size}_T{lookback_length}_speed15.npy', result)
np.save(f'1v3AttackDefend_g{grid_size1}{grid_size2}_T{lookback_length}_speed15.npy', result)

print(f"The value function has been saved successfully.")

# Record the time of whole process
end_time = time.time()
print(f"The end time is {end_time}")
print(f"The time of whole process is {end_time - start_time} seconds.")