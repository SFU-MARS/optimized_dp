import numpy as np
# Utility functions to initialize the problem
from Grid.GridProcessing import Grid
from Shapes.ShapesFunctions import *
# Specify the  file that includes dynamic systems
from dynamics.DubinsCar import *
# Plot options
from plot_options import *
# Solver core
from solver import *
import math

g = Grid(np.array([-3.0, -1.0, -math.pi]), np.array([3.0, 4.0, math.pi]), 3, np.array([160, 160, 160]), [2])
my_car = DubinsCar()

Initial_value_f = CylinderShape(g, [2], np.array([0.0, 1.0, 0.0, 0.0]), 0.70)
po = PlotOptions("3d_plot", [0,1,2], [])
eps = 0.00001
V_0 = TTRSolver(my_car, g, Initial_value_f, eps, po)
np.save("tt2_array.npy", V_0)
