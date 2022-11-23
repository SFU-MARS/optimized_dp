import numpy as np
from odp.Plots.plotting_utilities import plot_2d
from odp.Grid import Grid
from utilities import check, state_value
from odp.solver import HJSolver, computeSpatDerivArray
from MaximumMatching import MaxMatching

# initialize attackers and defenders

# initialize the maximum_matching logic

# apply control to the agents
# g must be the same as one in the ValueFunction.py
g = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))
value_function = np.load('1v1AttackDefend.npy')
# print(state_value(value_function, 0.5, 0.6, -0.3, 0.5))
# print(check(value_function, 0.5, 0.6, -0.3, 0.5, 45))
