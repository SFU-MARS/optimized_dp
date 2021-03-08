import numpy as np
# Utility functions to initialize the problem
from Grid.GridProcessing import Grid
from Shapes.ShapesFunctions import *
# Specify the  file that includes dynamic systems
from dynamics.DubinsCar4D import *
from dynamics.DubinsCapture import *
from dynamics.DubinsCar4D2 import *
# Plot options
from plot_options import *
# Solver core
from solver import HJSolver

import math

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""


# Scenario 1
g = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([40, 40, 40]), [2])

Initial_value_f = CylinderShape(g, [], np.zeros(3), 1)

# Look-back lenght and time step
lookback_length = 2.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
my_car = DubinsCapture()

po2 = PlotOptions("3d_plot", [0,1,2], [])

"""
Assign one of the following strings to `compMethod` to specify the characteristics of computation
"none" -> compute Backward Reachable Set
"minVWithV0" -> compute Backward Reachable Tube
"maxVWithVInit" -> compute max V over time
"minVWithVInit" compute min V over time
"""

# HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)
HJSolver(my_car, g, Initial_value_f, tau, "minVWithV0", po2)

# Second Scenario
g = Grid(np.array([-3.0, -1.0, 0.0, -math.pi]), np.array([3.0, 4.0, 4.0, math.pi]), 4, np.array([60, 60, 20, 36]), [3])

# Define my object
my_car = DubinsCar4D2()

# Use the grid to initialize initial value function
Initial_value_f = CylinderShape(g, [2,3], np.zeros(4), 1)

# Look-back lenght and time step
lookback_length = 1.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

po = PlotOptions("3d_plot", [0,1,3], [19])
HJSolver(my_car, g, Initial_value_f, tau, "minVWithV0", po)