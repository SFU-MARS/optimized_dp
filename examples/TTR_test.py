import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *
# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar
# Plot options
from odp.Plots import PlotOptions 
# Solver core

from odp.solver import HJSolver, TTRSolver, TTRSolver_Dev
import math

# Compute BRS only
g = Grid(minBounds=np.array([-3.0, -1.0, -math.pi]), maxBounds=np.array([3.0, 4.0, math.pi]),
         dims=3, pts_each_dim=np.array([80, 80, 80]), periodicDims=[2])
# Car is trying to reach the target
my_car = DubinsCar(uMode="min")

# Initialize target set as a cylinder
targeSet = CylinderShape(g, [2], np.array([0.0, 1.0, 0.0]), 0.70)

po = PlotOptions("set", plotDims=[0,1,2], slicesCut=[],
                min_isosurface=0, max_isosurface=0)

lookback_length = 1.5
t_step = 0.05
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

compMethod = { "TargetSetMode": "minVWithV0"}
accuracy = "low"
# correct_result = HJSolver(my_car, g, targeSet,
#                           tau, compMethod, po, accuracy)

# -------------------------------- ONE-SHOT TTR COMPUTATION ---------------------------------- #
# Car is trying to reach the target
my_car = DubinsCar(uMode="min")

# TTR Example 1
g = Grid(minBounds=np.array([-3.0, -1.0, -math.pi]), maxBounds=np.array([3.0, 4.0, math.pi]),
         dims=3, pts_each_dim=np.array([50, 50, 50]), periodicDims=[2])
targetSet = ShapeRectangle(g, [1.0, 2.0, -1000], [2.0, 3.0, 1000])


# # TTR Example 2 from the robut_utils
# g = Grid(minBounds=np.array([-6.5, -13, 0.0]), maxBounds=np.array([21.5, -1.5, 2*math.pi]),
#          dims=3, pts_each_dim=np.array([561, 231, 36]), periodicDims=[2])
# goal = np.array([-2.9, 4.6])
# targetSet = CylinderShape(g, [2], np.array(goal), 0.25)
# # obstacles = 

# Plot options
po = PlotOptions( "set", plotDims=[0,1,2], slicesCut=[],
                  min_isosurface=lookback_length, max_isosurface=lookback_length)


epsilon = 0.001

# First compute TTR set with the origional TTR solver
V_0 = TTRSolver(my_car, g, targetSet, epsilon, po)

## Hanyang: developing
#TODO: Test the TTR solver without obstacles
# V_0_Dev = TTRSolver_Dev(my_car, g, targeSet, epsilon, po)

# # Test the TTR solver with obstacles
obs = ShapeRectangle(g, [-1.0, 0.0, -1000], [0.0, 2.0, 1000])
V_0_Dev = TTRSolver_Dev(my_car, g, [targetSet, obs], epsilon, po)
# ## Compare the results with the original TTR solver
check_position1 = g.get_index((-0.5, -0.5, 1.57))
print(f"V_0 at position {check_position1}: {V_0[check_position1]}")
print(f"V_0_Dev at position {check_position1}: {V_0_Dev[check_position1]} \n")

check_position2 = g.get_index((-1.0, -1.0, 1.57))
print(f"V_0 at position {check_position2}: {V_0[check_position2]}")
print(f"V_0_Dev at position {check_position2}: {V_0_Dev[check_position2]} \n")

check_position3 = g.get_index((-2, 3, 0.0))
print(f"V_0 at position {check_position3}: {V_0[check_position3]}")
print(f"V_0_Dev at position {check_position3}: {V_0_Dev[check_position3]} \n")

