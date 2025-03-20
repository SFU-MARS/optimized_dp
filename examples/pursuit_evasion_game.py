import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCapture
# Plot options
from odp.Plots import PlotOptions, plot_isosurface, plot_valuefunction

# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import math

# Define grid
g = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([40, 40, 40]), [2])

# Implicit function for the initial value function
Initial_value_f = CylinderShape(g, [2], np.zeros(3), 1)

# Look-back length and time step of computation
lookback_length = 2.
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# uMode maximizing means avoiding capture, dMode minimizing means capturing
my_car = DubinsCapture(uMode="max", dMode="min")

# Specify how to plot the isosurface of the value function ( for higher-than-3-dimension arrays, which slices, indices
# we should plot if we plot at all )
po2 = PlotOptions(do_plot=True, plot_type="set", plotDims=[0,1,2],
                  slicesCut=[], save_fig=True, filename="test_pursuit_evasion.png")

"""
Assign one of the following strings to `TargetSetMode` to specify the characteristics of computation
"TargetSetMode":
{
"none" -> compute Backward Reachable Set,
"minVWithV0" -> min V with V0 (compute Backward Reachable Tube),
"maxVWithV0" -> max V with V0,
"maxVOverTime" -> compute max V over time,
"minVOverTime" -> compute min V over time,
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
compMethods = { "TargetSetMode": "minVWithV0"}
# HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)
result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, po2, saveAllTimeSteps=True )

last_time_step_result = result[..., 0]
# Compute spatial derivatives at every state
x_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=1, accuracy="low")
y_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=2, accuracy="low")
T_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=3, accuracy="low")

# Let's compute optimal control at some random idices
spat_deriv_vector = (x_derivative[10,20,30], y_derivative[10,20,30], T_derivative[10,20,30])
state_vector = (g.grid_points[0][10], g.grid_points[1][20], g.grid_points[2][30])

# Compute the optimal control
opt_ctrl = my_car.optCtrl_inPython(state_vector, spat_deriv_vector)

plot_isosurface(g, last_time_step_result, po2)