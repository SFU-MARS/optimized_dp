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
from Plots.plotting_utilities import *
# Solver core
from solver import HJSolver

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""

g = Grid(np.array([-5, -5, -np.pi]), np.array([5, 5, np.pi]), 3, np.array([41, 41, 41]), [2])
target_set = CylinderShape(g, [], np.zeros(3), 1)

tau_len = 200
tau = np.linspace(0, 10, tau_len)

uMode = "min"
dMode = "max"

dyn_sys = DubinsCapture()
po = PlotOptions(plot_type="3d_plot", dims_plot=[0,0,0], slices=[])

if __name__ in "__main__":
    V = HJSolver(dyn_sys, g, target_set, tau, "maxVWithVInit", po, save_all_t=True)
    np.save("V_with_time", V)
# for i in range(0, 20, 5):
#     po = PlotOptions("3d_plot", [0, 1, 3], [9, i])
#     print(f"time: {i}")
#     plot_isosurface(g, V, po)

# po = PlotOptions("3d_plot", [0, 1, 3], [9, 19])
# plot_isosurface(g, V, po)
