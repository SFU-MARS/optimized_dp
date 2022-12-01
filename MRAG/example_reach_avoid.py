import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *
from odp.Shapes.ShapesFunctions import ShapeRectangle1

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCapture
from odp.dynamics import DubinsCar4D2
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


##################################################### EXAMPLE 3 #####################################################

# Third scenario with Reach-Avoid set
g = Grid(np.array([-1.0, -1.0, -math.pi]), np.array([1.0, 1.0, math.pi]), 3, np.array([40, 40, 40]), [2])

# Reachable set
goal = CylinderShape(g, [2], np.array([0.5, 0.5, 0]), 0.2) # debug7
# goal = ShapeRectangle1(g, [0.6, 0.1, 0.0], [0.8, 0.3, 0.0], [2])  # debug8


# Avoid set
# obstacle = CylinderShape(g, [2], np.array([1.0, 1.0, 0.0]), 0.5)

# Avoid set, no constraint means inf
obstacle = CylinderShape(g, [2], np.array([0.0, 0.5, 0.0]), 0.3) # debug7
# obstacle = ShapeRectangle1(g, [-0.1, 0.30, 0.0], [0.1, 0.60, 0.0], [2])  # debug8

# Look-back length and time step
lookback_length = 20.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

my_car = DubinsCapture(uMode="min", dMode="max")

po2 = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0,1,2],
                  slicesCut=[])

"""
Assign one of the following strings to `TargetSetMode` to specify the characteristics of computation
"TargetSetMode":
{
"none" -> compute Backward Reachable Set,
"minVWithV0" -> min V with V0 (compute Backward Reachable Tube),
"maxVWithV0" -> max V with V0,
"maxVWithVInit" -> compute max V over time,
"minVWithVInit" -> compute min V over time,
"minVWithVTarget" -> min V with target set (if target set is different from initial V0)
"maxVWithVTarget" -> max V with target set (if target set is different from initial V0)
}

(optional)
Please specify this mode if you would like to add another target set, which can be an obstacle set
for solving a reach-avoid problem
"ObstacleSetMode":
{
"minVWithObstacle" -> min with obstacle set,
"maxVWithObstacle" -> max with obstacle set
}
"""

compMethods = { "TargetSetMode": "minVWithVTarget",
                "ObstacleSetMode": "maxVWithObstacle"}
# HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)
result = HJSolver(my_car, g, [goal, obstacle], tau, compMethods, po2, saveAllTimeSteps=True )
print(f'The shape of the value function is {result.shape} \n')
# save the value function
np.save('example_reach_avoid.npy', result)
