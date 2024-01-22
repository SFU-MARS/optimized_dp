import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *
# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar
# Plot options
from odp.Plots import PlotOptions 
# Solver core

from odp.solver import HJSolver, TTRSolver
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
correct_result = HJSolver(my_car, g, targeSet,
                          tau, compMethod, po, accuracy)

# -------------------------------- ONE-SHOT TTR COMPUTATION ---------------------------------- #
g = Grid(minBounds=np.array([-3.0, -1.0, -math.pi]), maxBounds=np.array([3.0, 4.0, math.pi]),
         dims=3, pts_each_dim=np.array([50, 50, 50]), periodicDims=[2])
# Car is trying to reach the target
my_car = DubinsCar(uMode="min")

# Initialize target set as a cylinder
targetSet = CylinderShape(g, [2], np.array([0.0, 1.0, 0.0]), 0.70)
po = PlotOptions( "set", plotDims=[0,1,2], slicesCut=[],
                  min_isosurface=lookback_length, max_isosurface=lookback_length)

# First compute TTR set
epsilon = 0.001
V_0 = TTRSolver(my_car, g, targetSet, epsilon, po)

#np.save("tt2_array.npy", V_0)
