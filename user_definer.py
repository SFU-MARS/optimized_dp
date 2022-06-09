import numpy as np
# Utility functions to initialize the problem
from Grid.GridProcessing import Grid
from Shapes.ShapesFunctions import *
# Specify the  file that includes dynamic systems
from dynamics.DubinsCar4D import *
from dynamics.DubinsCapture import *
from dynamics.DubinsCar4D2 import *
from dynamics.DubinsCar5DAvoid import *
# Plot options
from plot_options import *
# Solver core
from solver import HJSolver
import plot_contour_5D

import math

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""

# Scenario 1
g = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([150, 150, 150]), [2])

Initial_value_f = CylinderShape(g, [], np.zeros(3), 1)

# Look-back length and time step
lookback_length = 2.8
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
my_car = DubinsCapture()

po2 = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0,1,2],
                   slicesCut=[])

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
Please specify this mode if you would like to solve a reach-avoid problem
"TargetSetMode":
{
"min" -> min with target set,
"max" -> max with taget set
}
"""

compMethods = { "PrevSetsMode": "minVWithV0"}
# HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)
result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, po2, saveAllTimeSteps=False)

# for i in range(6):
#     plot_contour_5D.plot_contour(result, idx1=i * 4)
#
# # Second Scenario
# g = Grid(np.array([-3.0, -1.0, 0.0, -math.pi]), np.array([3.0, 4.0, 4.0, math.pi]), 4, np.array([60, 60, 20, 36]), [3])
#
# # Define my object
# my_car = DubinsCar4D2()
#
# # Use the grid to initialize initial value function
# Initial_value_f = CylinderShape(g, [2,3], np.zeros(4), 1)
#
# # Look-back lenght and time step
# lookback_length = 1.0
# t_step = 0.05
#
# small_number = 1e-5
#
# tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
#
# po = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0,1,3],
#                   slicesCut=[19, ])
#
# compMethods = { "PrevSetsMode": "minVWithV0"}
# result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, po, saveAllTimeSteps=True, untilConvergent=True)
# print(result.shape)

# Third Scenario
# g = Grid(np.array([-15.0, -15.0, 0, 0, 0]), np.array([15, 15, 2*math.pi, 5.5, 5.5]), 5, np.array([51, 51, 21, 21, 21]), [2])
#
# # Define my object
# my_car = DubinsCar5DAvoid(x=[0,0,0,0,0], u_theta_max = math.pi/2, u_v_max=3, d_theta_max=math.pi/2, d_v_max=3, uMode="max", dMode="min")
#
# Initial_value_f = np.minimum(CylinderShape(g, [3,4,5], np.zeros(5), 2), Lower_Half_Space(g, 3, 0.5))
# Initial_value_f = np.minimum(Initial_value_f, Upper_Half_Space(g, 3, 5.0))
#
# # Use the grid to initialize initial value function
# Initial_value_f = CylinderShape(g, [2,3], np.zeros(5), 1)
#
# # Look-back lenght and time step
# lookback_length = 0.1
# t_step = 0.05
#
# small_number = 1e-5
#
# tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
#
# po = PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0,1,2],
#                   slicesCut=[19, 19])
#
# compMethods = { "PrevSetsMode": "minVWithV0"}
# result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, po, saveAllTimeSteps=False)


# 5D case code
# g = Grid(np.array([-0.35, -0.35, -math.pi, -0.25, -3.75]), np.array([0.35, 0.35, math.pi, 0.25, 3.75]), 5, np.array([40, 40, 40, 40, 40]), [2])
#
# # Define my object
# my_car = DubinsCar5DAvoid(x=[0,0,0,0,0], u_theta_max = 1, u_v_max=1, d_theta_max=1, d_v_max=1, uMode="max", dMode="min")
#
# Initial_value_f = CylinderShape(g, [2,3,4], np.zeros(5), 1)
# # Initial_value_f = np.minimum(Initial_value_f, Upper_Half_Space(g, 3, 5.0))
#
# # Look-back lenght and time step
# lookback_length = 0.1
# t_step = 0.05
#
# small_number = 1e-5
#
# tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
#
# po = PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0,1,2],
#                  slicesCut=[19, 19])
#
# compMethods = { "PrevSetsMode": "minVWithV0"}
#result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, po, saveAllTimeSteps=False, accuracy="high")
