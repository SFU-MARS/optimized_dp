import os
import gc
import time
import psutil
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *
# Specify the  file that includes dynamic systems, AttackerDefender4D
from MRAG.AttackerDefender1v2 import AttackerDefender1v2 
# Plot options
from odp.Plots import PlotOptions
from odp.Plots.plotting_utilities import plot_2d, plot_isosurface
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray

""" USER INTERFACES
- 1. Define grid
- 2. Instantiate the dynamics of the agent
- 3. Generate initial values for grid using shape functions
- 4. Time length for computations
- 5. Initialize plotting option
- 6. Call HJSolver function
"""

##################################################### EXAMPLE 5 1 vs. 2 AttackerDefender ####################################
# Record the time of whole process
start_time = time.time()

# 1. Define grid
grid_size = 30

grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
             6, np.array([grid_size, grid_size, grid_size, grid_size, grid_size, grid_size])) 
process = psutil.Process(os.getpid())
print("1. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

# 2. Instantiate the dynamics of the agent
agents_1v2 = AttackerDefender1v2(uMode="min", dMode="max")  

# 3. Avoid set, no constraint means inf
obs1_attack = ShapeRectangle(grids, [-0.1, -1.0, -1000, -1000, -1000, -1000], [0.1, -0.3, 1000, 1000, 1000, 1000])  # attacker stuck in obs1
obs1_attack = np.array(obs1_attack, dtype='float32')

obs2_attack = ShapeRectangle(grids, [-0.1, 0.30, -1000, -1000, -1000, -1000], [0.1, 0.60, 1000, 1000, 1000, 1000])  # attacker stuck in obs2
obs2_attack = np.array(obs2_attack, dtype='float32')

obs_attack = np.minimum(obs1_attack, obs2_attack)  # the union of being stuck in obs1 and obs2
del obs1_attack
del obs2_attack

obs3_capture1 = agents_1v2.capture_set1(grids, 0.1, "capture")  
obs3_capture1 = np.array(obs3_capture1, dtype='float32')

obs3_capture2 = agents_1v2.capture_set2(grids, 0.1, "capture")  
obs3_capture2 = np.array(obs3_capture2, dtype='float32')

obs3_capture = np.minimum(obs3_capture1, obs3_capture2)  # the union of being captured by defender 1 and 2
obs3_capture = np.array(obs3_capture, dtype='float32')
del obs3_capture1
del obs3_capture2

avoid_set = np.minimum(obs3_capture, obs_attack) # the union of being stuck in obs or being captured by defender
avoid_set = np.array(avoid_set, dtype='float32')
del obs_attack
del obs3_capture
print("2. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

# 4. Reach set, no constraint means inf
goal1_destination = ShapeRectangle(grids, [0.6, 0.1, -1000, -1000, -1000, -1000], [0.8, 0.3, 1000, 1000, 1000, 1000])  # attacker arrives target
goal2_escape1 = agents_1v2.capture_set1(grids, 0.1, "escape")  # attacker escape from defender
goal2_escape2 = agents_1v2.capture_set2(grids, 0.1, "escape")  # attacker escape from defender
goal2_escape = np.maximum(goal2_escape1, goal2_escape2)  # the intersection of escaping from defender 1 and 2
goal2_escape = np.array(goal2_escape, dtype='float32')

# Defender 1 gets stuck in obs 
obs1_defend1 = ShapeRectangle(grids, [-1000, -1000, -0.1, -1.0, -1000, -1000], [1000, 1000, 0.1, -0.3, 1000, 1000])  # defender 1 stuck in obs1
obs2_defend1 = ShapeRectangle(grids, [-1000, -1000, -0.1, 0.30, -1000, -1000], [1000, 1000, 0.1, 0.60, 1000, 1000])  # defender 1 stuck in obs2
obs_defender1 = np.minimum(obs1_defend1, obs2_defend1)  # the union of defender 1 stuck in obs1 and obs2
obs_defender1 = np.array(obs_defender1, dtype='float32')
del obs1_defend1
del obs2_defend1

# Defender 2 gets stuck in obs 
obs1_defend2 = ShapeRectangle(grids, [-1000, -1000, -1000, -1000, -0.1, -1.0], [1000, 1000, 1000, 1000, 0.1, -0.3])  # defender 2 stuck in obs1
obs2_defend2 = ShapeRectangle(grids, [-1000, -1000, -1000, -1000, -0.1, 0.30], [1000, 1000, 1000, 1000, 0.1, 0.60])  # defender 2 stuck in obs2
obs_defender2 = np.minimum(obs1_defend2, obs2_defend2)  # the union of defender 2 stuck in obs1 and obs2
obs_defender2 = np.array(obs_defender2, dtype='float32')
del obs1_defend2
del obs2_defend2 

obs_defends = np.minimum(obs_defender1, obs_defender2)  # Michael: change this to union 
obs_defends = np.array(obs_defends, dtype='float32')

reach_set = np.minimum(obs_defends, np.maximum(goal1_destination, goal2_escape))
del obs_defends
del goal1_destination
del goal2_escape
gc.collect()
process = psutil.Process(os.getpid())
print("3. Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

# 4. Look-back length and time step
lookback_length = 4.5  # the same as 2014Mo
t_step = 0.025

# Actual calculation process, needs to add new plot function to draw a 2D figure
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# while plotting make sure the len(slicesCut) + len(plotDims) = grid.dims
po = PlotOptions(do_plot=False, plot_type="2d_plot", plotDims=[0, 1], slicesCut=[22, 22])

# In this example, we compute a Reach-Avoid Tube
compMethods = {"TargetSetMode": "minVWithVTarget", "ObstacleSetMode": "maxVWithObstacle"} # original one
# compMethods = {"TargetSetMode": "minVWithVTarget"}
solve_start_time = time.time()

result = HJSolver(agents_1v2, grids, [reach_set, avoid_set], tau, compMethods, po, saveAllTimeSteps=None) # original one
# result = HJSolver(my_2agents, g, avoid_set, tau, compMethods, po, saveAllTimeSteps=True)
process = psutil.Process(os.getpid())
print(f"The CPU memory used during the calculation of the value function is {process.memory_info().rss/(1024 ** 3): .2f} GB.")  # in bytes

solve_end_time = time.time()
print(f'The shape of the value function is {result.shape} \n')
print(f"The size of the value function is {result.nbytes / (1024 ** 3): .2f} GB or {result.nbytes/(1024 ** 2)} MB.")
print(f"The time of solving HJ is {solve_end_time - solve_start_time} seconds.")
print(f'The shape of the value function is {result.shape} \n')
# save the value function
# np.save('/localhome/hha160/optimized_dp/MRAG/1v1AttackDefend_speed15.npy', result)  # grid = 45
# np.save(f'1v2AttackDefend_g{grid_size}_speed15.npy', result)  # grid = 30
np.save(f'1v2AttackDefend_Michael_speed15.npy', result)

print(f"The value function has been saved successfully.")
# Record the time of whole process
end_time = time.time()
print(f"The time of whole process is {end_time - start_time} seconds.")