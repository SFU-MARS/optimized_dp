import numpy as np

# Utility functions to initialize the problem
from Grid.GridProcessing import Grid
from Shapes.ShapesFunctions import *

# Specify the  file that includes dynamic systems
from dynamics.DubinsCar4D import *
from dynamics.DubinsCapture import *
from dynamics.DubinsCar4D2 import *
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


g = Grid(
    np.array([-4.5, -4.5, -np.pi]),
    np.array([4.5, 4.5, np.pi]),
    3,
    np.array([80, 80, 40]),
    [2],
)

car_r = 0.5
# Implicit function is a spherical shape - check out CylinderShape API
Initial_value_f = ShapeRectangle(
    g,
    # np.array([-4.5 - 0.5, -0.5 + 0.5, -np.inf]),
    np.array([-4.5, -0.5 - car_r, -np.inf]),
    np.array([-4.5 + 6.5, -0.5 + 1.2 + car_r, np.inf]),
)

Initial_value_f = Union(
    Initial_value_f,
    Union(Lower_Half_Space(g, 0, -4.5 + car_r), Upper_Half_Space(g, 0, 4.5 - car_r)),
)

Initial_value_f = Union(
    Initial_value_f,
    Union(Lower_Half_Space(g, 1, -4.5 + car_r), Upper_Half_Space(g, 1, 4.5 - car_r)),
)
# Look-back length and time step of computation
lookback_length = 2.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# Specify the dynamical object. DubinsCapture() has been declared in dynamics/
# Control uMode is maximizing, meaning that we're avoiding the initial target set
my_car = DubinsCar(uMode="max", dMode="min")

# Specify how to plot the isosurface of the value function ( for higher-than-3-dimension arrays, which slices, indices
# we should plot if we plot at all )
po2 = PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[])

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

# In this example, we compute a Backward Reachable Tube
compMethods = {"TargetSetMode": "minVWithVInit"}
# HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)

result = HJSolver(
    my_car, g, Initial_value_f, tau, compMethods, po2, saveAllTimeSteps=False
)

lookback_length = 0.5
t_step = 0.05

small_number = 1e-5
tau2 = np.arange(start=0, stop=lookback_length + small_number, step=t_step)


my_car = DubinsCar(uMode="min", dMode="max")
compMethods = {"TargetSetMode": "maxVWithVInit"}
result = HJSolver(my_car, g, result, tau2, compMethods, po2, saveAllTimeSteps=False)

# print(result.shape)
np.save("max_over_min_brt.npy", result)

# ##################################################### EXAMPLE 3 #####################################################

# # Third scenario with Reach-Avoid set
# g = Grid(
#     np.array([-4.0, -4.0, -math.pi]),
#     np.array([4.0, 4.0, math.pi]),
#     3,
#     np.array([40, 40, 40]),
#     [2],
# )

# # Reachable set
# goal = CylinderShape(g, [2], np.zeros(3), 0.5)

# # Avoid set
# obstacle = CylinderShape(g, [2], np.array([1.0, 1.0, 0.0]), 0.5)

# # Look-back length and time step
# lookback_length = 1.5
# t_step = 0.05

# small_number = 1e-5
# tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# my_car = DubinsCapture(uMode="min", dMode="max")

# po2 = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[])

# """
# Assign one of the following strings to `TargetSetMode` to specify the characteristics of computation
# "TargetSetMode":
# {
# "none" -> compute Backward Reachable Set,
# "minVWithV0" -> min V with V0 (compute Backward Reachable Tube),
# "maxVWithV0" -> max V with V0,
# "maxVWithVInit" -> compute max V over time,
# "minVWithVInit" -> compute min V over time,
# "minVWithVTarget" -> min V with target set (if target set is different from initial V0)
# "maxVWithVTarget" -> max V with target set (if target set is different from initial V0)
# }

# (optional)
# Please specify this mode if you would like to add another target set, which can be an obstacle set
# for solving a reach-avoid problem
# "ObstacleSetMode":
# {
# "minVWithObstacle" -> min with obstacle set,
# "maxVWithObstacle" -> max with obstacle set
# }
# """

# compMethods = {
#     "TargetSetMode": "minVWithVTarget",
#     "ObstacleSetMode": "maxVWithObstacle",
# }
# # HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)
# result = HJSolver(
#     my_car, g, [goal, obstacle], tau, compMethods, po2, saveAllTimeSteps=True
# )
