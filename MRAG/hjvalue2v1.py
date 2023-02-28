import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *
# Specify the  file that includes dynamic systems
from MRAG.AttackerDefender2v1 import AttackerDefender2v1 
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

##################################################### EXAMPLE 5 2v1AttackerDefender ####################################

grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([45, 45, 45, 45, 45, 45])) # original 45

# Define my object dynamics
agents_2v1 = AttackerDefender2v1(uMode="min", dMode="max")  # 2v1 (6 dim dynamics)

# Avoid set, no constraint means inf
obs1_attack = ShapeRectangle(grids, [-0.1, -1.0, -1000, -1000, -1000, -1000], [0.1, -0.3, 1000, 1000, 1000, 1000])  # a1 get stuck in the obs1
obs2_attack = ShapeRectangle(grids, [-0.1, 0.30, -1000, -1000], [0.1, 0.60, 1000, 1000])  # attacker stuck in obs2
obs3_capture = agents_2v1.capture_set(grids, 0.1, "capture")  # attacker being captured by defender, try different radius
avoid_set = np.minimum(obs3_capture, np.minimum(obs1_attack, obs2_attack)) # original

# Reach set, run and see what it is!
goal1_destination = ShapeRectangle(grids, [0.6, 0.1, -1000, -1000], [0.8, 0.3, 1000, 1000])  # attacker arrives target
goal2_escape = agents_2v1.capture_set(grids, 0.1, "escape")  # attacker escape from defender
obs1_defend = ShapeRectangle(grids, [-1000, -1000, -0.1, -1000], [1000, 1000, 0.1, -0.3])  # defender stuck in obs1
obs2_defend = ShapeRectangle(grids, [-1000, -1000, -0.1, 0.30], [1000, 1000, 0.1, 0.60])  # defender stuck in obs2
reach_set = np.minimum(np.maximum(goal1_destination, goal2_escape), np.minimum(obs1_defend, obs2_defend)) # original

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
result = HJSolver(agents_2v1, grids, [reach_set, avoid_set], tau, compMethods, po, saveAllTimeSteps=True) # original one

print(f'The shape of the value function is {result.shape} \n')
# save the value function
np.save('2v1AttackDefend.npy', result)
