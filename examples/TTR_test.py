import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *
# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar, Plane2D, DubinsCar4D
# Plot options
from odp.Plots import PlotOptions 
from odp.Plots.plotting_utilities import plot_isosurface
# Solver core

from odp.solver import HJSolver, TTRSolver, TTRSolver_Dev
import math

# -------------------------------- ONE-SHOT TTR COMPUTATION ---------------------------------- #

# # TTR Example 1
# g = Grid(minBounds=np.array([-3.0, -1.0, -math.pi]), maxBounds=np.array([3.0, 4.0, math.pi]),
#          dims=3, pts_each_dim=np.array([50, 50, 50]), periodicDims=[2])
# targetSet = ShapeRectangle(g, [1.0, 2.0, -1000], [2.0, 3.0, 1000])
# my_car = DubinsCar(uMode="min")
# # First compute TTR set with the origional TTR solver
# # V_0 = TTRSolver(my_car, g, targetSet, epsilon, po)

# ## Hanyang: developing
# # # Test the TTR solver without obstacles
# # V_0_Dev = TTRSolver_Dev(my_car, g, targetSet, epsilon, po)

# # # Test the TTR solver with obstacles
# # obs = ShapeRectangle(g, [-1.0, 0.0, -1000], [0.0, 2.0, 1000])
# # obs = ShapeRectangle(g, [-2.0, 0.0, -1000], [0.0, 1.0, 1000])
# obs_goal = np.array([-0.5, 0.5])
# obs = CylinderShape(g, [2], np.array(obs_goal), 0.5)
# po = PlotOptions( plot_type="value", plotDims=[0,1], slicesCut=[25])
# epsilon = 0.001
# V_0_Dev = TTRSolver_Dev(my_car, g, [targetSet, obs], epsilon, po)


# # TTR Example 2 from the robut_utils
# g = Grid(minBounds=np.array([-6.5, -13, 0.0]), maxBounds=np.array([21.5, -1.5, 2*math.pi]),
#          dims=3, pts_each_dim=np.array([561, 231, 36]), periodicDims=[2])
# goal = np.array([-2.9, 4.6])
# targetSet = CylinderShape(g, [2], np.array(goal), 0.25)
# # obstacles = 


# TTR Example 3: 2D SIG works
g = Grid(minBounds=np.array([-3.0, -1.0]), maxBounds=np.array([3.0, 4.0]),
         dims=2, pts_each_dim=np.array([50, 50]), periodicDims=[])
my_sig = Plane2D(uMode="min")
targetSet = ShapeRectangle(g, [1.0, 2.0], [2.0, 3.0])
obs = ShapeRectangle(g, [-2.0, 0.0], [0.0, 1.0])
po = PlotOptions( plot_type="value", plotDims=[0,1], slicesCut=[])
epsilon = 0.001
V_0_Dev = TTRSolver_Dev(my_sig, g, [targetSet, obs], epsilon, po)


# # TTR Example 4: 4D DubinsCar4D
# my_car4D = DubinsCar4D(uMode="min")
# g = Grid(minBounds=np.array([-3.0, -1.0, -0.5, -math.pi]), maxBounds=np.array([3.0, 4.0, 0.5, math.pi]),
#          dims=4, pts_each_dim=np.array([50, 50, 50, 50]), periodicDims=[3])
# targetSet = ShapeRectangle(g, [1.0, 2.0, -100, -100], [2.0, 3.0, 100, 100])
# obs = ShapeRectangle(g, [-2.0, 0.0, -100, -100], [0.0, 1.0, 100, 100])
# po = PlotOptions( plot_type="value", plotDims=[0,1], slicesCut=[25, 25])
# epsilon = 0.001
# V_0_Dev = TTRSolver_Dev(my_car4D, g, [targetSet, obs], epsilon, po)


