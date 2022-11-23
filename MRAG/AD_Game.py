import numpy as np
from odp.Plots.plotting_utilities import plot_2d
from odp.Grid import Grid
from odp.solver import HJSolver, computeSpatDerivArray
from MaximumMatching import MaxMatching

# initialize attackers and defenders

# initialize the maximum_matching logic

# apply control to the agents
# g must be the same as one in the ValueFunction.py
g = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([31, 31, 31, 31]))
# value_function = np.load('1v1AttackDefend.npy')
value_function = np.load('result.npy')
