import os
import time
import psutil
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

""" USER INTERFACES
- 1. Initialize the grids
- 2. Initialize the dynamics
- 3. Instruct the avoid set and reach set
- 4. Set the look-back length and time step
- 5. Call HJSolver function
- 6. Save the value function
"""

##################################################### EXAMPLE 4 1v1AttackerDefender ####################################
# Record the time of whole process
start_time = time.time()

# 1. Initialize the grids
# grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([25, 25, 25, 25]))
# grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([27, 27, 27, 27]))
# grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([29, 29, 29, 29]))
# grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([31, 31, 31, 31]))
# grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([33, 33, 33, 33]))
# grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([35, 35, 35, 35]))
grid_size = 35
speed_d = 1.5
grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([grid_size, grid_size, grid_size, grid_size]))

# 2. Initialize the dynamics
agents_1v1 = AttackerDefender1v1(uMode="min", dMode="max", speed_a=1.0, speed_d=speed_d)  # 1v1 (4 dims dynamics)

# 3. Instruct the avoid set and reach set
# 3.1 Avoid set, no constraint means inf
obs1_attack = ShapeRectangle(grids, [-0.1, -1.0, -1000, -1000], [0.1, -0.3, 1000, 1000])  # attacker stuck in obs1
obs2_attack = ShapeRectangle(grids, [-0.1, 0.30, -1000, -1000], [0.1, 0.60, 1000, 1000])  # attacker stuck in obs2
obs3_capture = agents_1v1.capture_set(grids, 0.1, "capture")  # attacker being captured by defender
avoid_set = np.minimum(obs3_capture, np.minimum(obs1_attack, obs2_attack)) # original

# 3.2 Reach set, run and see what it is!
goal1_destination = ShapeRectangle(grids, [0.6, 0.1, -1000, -1000], [0.8, 0.3, 1000, 1000])  # attacker arrives target
goal2_escape = agents_1v1.capture_set(grids, 0.1, "escape")  # attacker escape from defender
obs1_defend = ShapeRectangle(grids, [-1000, -1000, -0.1, -1000], [1000, 1000, 0.1, -0.3])  # defender stuck in obs1
obs2_defend = ShapeRectangle(grids, [-1000, -1000, -0.1, 0.30], [1000, 1000, 0.1, 0.60])  # defender stuck in obs2
reach_set = np.minimum(np.maximum(goal1_destination, goal2_escape), np.minimum(obs1_defend, obs2_defend)) # original

# 4. Set the look-back length and time step
lookback_length = 10  # the same as 2014Mo
t_step = 0.025

# Actual calculation process, needs to add new plot function to draw a 2D figure
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# while plotting make sure the len(slicesCut) + len(plotDims) = grid.dims
po = PlotOptions(do_plot=False, plot_type="2d_plot", plotDims=[0, 1], slicesCut=[22, 22])

# 5. Call HJSolver function
compMethods = {"TargetSetMode": "minVWithVTarget", "ObstacleSetMode": "maxVWithObstacle"} # original one
# compMethods = {"TargetSetMode": "minVWithVTarget"}
solve_start_time = time.time()

result = HJSolver(agents_1v1, grids, [reach_set, avoid_set], tau, compMethods, po, saveAllTimeSteps=None) # original one
# result = HJSolver(my_2agents, g, avoid_set, tau, compMethods, po, saveAllTimeSteps=True)
process = psutil.Process(os.getpid())
print(f"The CPU memory used during the calculation of the value function is {process.memory_info().rss/1e9: .2f} GB.")  # in bytes

solve_end_time = time.time()
print(f'The shape of the value function is {result.shape} \n')
print(f"The size of the value function is {result.nbytes / 1e9: .2f} GB or {result.nbytes/(1e6)} MB.")
print(f"The time of solving HJ is {solve_end_time - solve_start_time} seconds.")
print(f'The shape of the value function is {result.shape} \n')

# 6. Save the value function
# np.save('/localhome/hha160/optimized_dp/MRAG/1v1AttackDefend_speed15.npy', result)  # grid = 45
# np.save('1v1AttackDefend_g25_speed15.npy', result)  # grid = 25
# np.save('1v1AttackDefend_g27_speed15.npy', result)  # grid = 27
# np.save('1v1AttackDefend_g29_speed15.npy', result)  # grid = 29
# np.save('1v1AttackDefend_g31_speed15.npy', result)  # grid = 31
# np.save('1v1AttackDefend_g33_speed15.npy', result)  # grid = 33
# np.save('1v1AttackDefend_g35_speed15.npy', result)  # grid = 35
np.save(f'MRAG/1v1AttackDefend_g{grid_size}_dspeed{speed_d}.npy', result) 
print("The value function has been saved successfully.")

# Record the time of whole process
end_time = time.time()
print(f"The time of whole process is {end_time - start_time} seconds.")