# import imp
import datetime
import numpy as np
# Utility functions to initialize the problem
from odp.grid import *
from odp.ShapesFunctions import *

# Specify the  file that includes dynamic systems
import odp.dynamics as odp_dyn

# Plot options
from odp.plot import PlotOptions
# from odp.Plots import plot_isosurface, plot_valuefunction

# Solver core
from odp.new_solver import HJSolver

import math

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""

##################################################### EXAMPLE 1 #####################################################

g = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([100, 100, 100]), [2])

# Implicit function is a spherical shape - check out CylinderShape API
Initial_value_f = CylinderShape(g, [2], np.zeros(3), 1)

# Look-back length and time step of computation
lookback_length = 3.
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# Specify the dynamical object. DubinsCapture() has been declared in dynamics/
# Control uMode is maximizing, meaning that we're avoiding the initial target set
pursuit_evasion_sys = odp_dyn.DubinsCapture(uMode="max", dMode="min")


# Specify how to plot the isosurface of the value function ( for higher-than-3-dimension arrays, which slices, indices
# we should plot if we plot at all )
# po2 = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1,2],
#                  slicesCut=[], save_fig=True, filename="test.png")

po2 = PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0,1,2],
                 slicesCut=[])


# In this example, we compute a Backward Reachable Tube
compMethods = { "TargetSetMode": "minVWithV0"}

# Initialize the solver first
pursuit_evasion_solver = HJSolver(model=pursuit_evasion_sys,
                                grid=g,
                                interactive=True,
                                accuracy='low',)

result = pursuit_evasion_solver(tau=tau,
                       target=Initial_value_f,
                       target_mode="min",
                       constraint=None)