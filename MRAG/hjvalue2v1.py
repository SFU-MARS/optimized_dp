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

grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([45, 45, 45, 45, 45, 45])) # original 45, on mars-14 20 is the upper bound

# Define my object dynamics
agents_2v1 = AttackerDefender2v1(uMode="min", dMode="max")  # 2v1 (6 dim dynamics)

# Avoid set, no constraint means inf
obs1_a1 = ShapeRectangle(grids, [-0.1, -1.0, -1000, -1000, -1000, -1000], [0.1, -0.3, 1000, 1000, 1000, 1000])  # a1 get stuck in the obs1
obs2_a1 = ShapeRectangle(grids, [-0.1, 0.30, -1000, -1000, -1000, -1000], [0.1, 0.60, 1000, 1000, 1000, 1000])  # a1 get stuck in the obs2
obs1_a2 = ShapeRectangle(grids, [-1000, -1000, -0.1, -1.0, -1000, -1000], [1000, 1000, 0.1, -0.3, 1000, 1000])  # a2 get stuck in the obs1
obs2_a2 = ShapeRectangle(grids, [-1000, -1000, -0.1, 0.30, -1000, -1000], [1000, 1000, 0.1, 0.60, 1000, 1000])  # a2 get stuck in the obs2
capture_a1 = agents_2v1.capture_set1(grids, 0.1, "capture")  # a1 is captured
capture_a2 = agents_2v1.capture_set2(grids, 0.1, "capture")  # a1 is captured
avoid_set = np.minimum(np.minimum(np.minimum(obs1_a2, obs2_a2), np.minimum(obs1_a1, obs2_a1)), np.minimum(capture_a1, capture_a2)) # is order necessary?
# debug


# Reach set, run and see what it is!
goal1_destination = ShapeRectangle(grids, [0.6, 0.1, 0.6, 0.1, -1000, -1000], [0.8, 0.3, 0.8, 0.3, 1000, 1000])  # a1 and a2 both arrive the goal
escape_a1 = agents_2v1.capture_set1(grids, 0.1, "escape")  # a1 escape
escape_a2 = agents_2v1.capture_set2(grids, 0.1, "escape")  # a2 escape
obs1_defend = ShapeRectangle(grids, [-1000, -1000, -1000, -1000, -0.1, -1000], [1000, 1000, 1000, 1000, 0.1, -0.3])  # defender stuck in obs1
obs2_defend = ShapeRectangle(grids, [-1000, -1000, -1000, -1000, -0.1, 0.30], [1000, 1000, 1000, 1000, 0.1, 0.60])  # defender stuck in obs2
reach_set = np.minimum(np.maximum(np.maximum(goal1_destination, escape_a1), np.maximum(goal1_destination, escape_a2)),  
                       np.minimum(obs1_defend, obs2_defend)) 

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
np.save('MRAG/2v1AttackDefend.npy', result)
