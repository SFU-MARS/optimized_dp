import imp
import os
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
lookback_length = 4.
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# uMode maximizing means avoiding capture, dMode minimizing means capturing
my_car = DubinsCapture(uMode="max", dMode="min")

# Specify how to plot the isosurface of the value function ( for higher-than-3-dimension arrays, which slices, indices
# we should plot if we plot at all )
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
po2 = PlotOptions(do_plot=True, plot_type="set", plotDims=[0,1,2],
                  slicesCut=[], save_fig=True, filename=f"{current_directory}/test_pursuit_evasion.png")

# In this example, we compute a Backward Reachable Tube
compMethods = { "TargetSetMode": "minVWithV0"}
# HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)
saveAllTimeSteps = True
result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, po2, saveAllTimeSteps=saveAllTimeSteps)

np.save(f"{current_directory}/hj_values/pursuit_evasion3d_{saveAllTimeSteps}AllTimeSteps_{lookback_length}.npy", result)
print(f"********** HJ value function is saved at {current_directory}/hj_values/pursuit_evasion3d_{saveAllTimeSteps}AllTimeSteps_{lookback_length}.npy. **********")
# plot_isosurface(g, result, po2)

# if saveAllTimeSteps:
#     last_time_step_result = result[..., 0]
#     # Compute spatial derivatives at every state
#     x_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=1, accuracy="low")
#     y_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=2, accuracy="low")
#     T_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=3, accuracy="low")

#     # Let's compute optimal control at some random idices
#     spat_deriv_vector = (x_derivative[10,20,30], y_derivative[10,20,30], T_derivative[10,20,30])
#     state_vector = (g.grid_points[0][10], g.grid_points[1][20], g.grid_points[2][30])

#     # Compute the optimal control
#     opt_ctrl = my_car.optCtrl_inPython(state_vector, spat_deriv_vector)

#     plot_isosurface(g, last_time_step_result, po2)