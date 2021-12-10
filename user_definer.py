import numpy as np
# Utility functions to initialize the problem
from Grid.GridProcessing import Grid
from Shapes.ShapesFunctions import *
# Specify the  file that includes dynamic systems
from dynamics.DubinsCar import *
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
g = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([101, 101, 101]), [2])
r = 0.75 + 0.4
Initial_value_f = CylinderShape(g, [2], np.zeros(3), r) # rad of obstacle + robot radius

# Look-back lenght and time step
lookback_length = 2.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
my_car = DubinsCar(uMode='max', dMode='min')

po = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0,1,2], slicesCut=[])

"""
Assign one of the following strings to `PrevSetMode` to specify the characteristics of computation
"PrevSetMode":
{
"none" -> compute Backward Reachable Set, 
"minVWithV0" -> compute Backward Reachable Tube,
"maxVWithVInit" -> compute max V over time,
"minVWithVInit" compute min V over time,
}

(optional)
"TargetSetMode":
{
"min" -> min with target set,
"max" -> max with taget set
}
"""

compMethods = { "PrevSetsMode": "minVWithV0"}
V = HJSolver(my_car, g, Initial_value_f, tau, compMethods, po)
np.save(f'V_r{r}_grid101', V)