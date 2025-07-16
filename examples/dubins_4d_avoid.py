import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar4D2
# Plot options
from odp.Plots import PlotOptions, visualize_plots

# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import math

# Define grid
g = Grid(np.array([-3.0, -1.0, 0.0, -math.pi]), np.array([3.0, 4.0, 4.0, math.pi]), 4, np.array([60, 60, 20, 36]), [3])

# Define my object
my_car = DubinsCar4D2(uMode="max", dMode="min")

# Use the grid to initialize initial value function
Initial_value_f = CylinderShape(g, [2,3], np.array([0., 2., 0., 0.]), 0.8)

# Look-back length and time step
lookback_length = 1.5
t_step = 0.05

small_number = 1e-5

tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# In this example, we compute a Backward Reachable Tube
compMethods = { "TargetSetMode": "minVWithV0"}
result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, saveAllTimeSteps=True, accuracy="medium")

last_time_step_result = result[..., 0]

# Compute spatial derivatives at every state
x_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=1, accuracy="medium")
y_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=2, accuracy="medium")
v_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=3, accuracy="medium")
T_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=4, accuracy="medium")

# Let's compute optimal control at some random indices
spat_deriv_vector = (x_derivative[10,20,15,15], y_derivative[10,20,15,15],
                     v_derivative[10,20,15,15], T_derivative[10,20,15,15])

# Compute the optimal control
opt_a, opt_w = my_car.optCtrl_inPython(spat_deriv_vector)

# Visualize the results
po = PlotOptions(do_plot=True, plot_type="set", plotDims=[0,1,3],
                  slicesCut=[10], save_fig=True, filename="test_obs_avoid.png")
visualize_plots(result, g, po)