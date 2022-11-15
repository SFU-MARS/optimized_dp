import numpy as np
from odp.Plots.plotting_utilities import plot_2d
from odp.Grid import Grid


# plot_2d figure
# g must be the same as one in the ValueFunction.py
g = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([31, 31, 31, 31]))
value_function = np.load('1v1AttackDefend.npy')
V_2D = value_function[:, :, 29, 29, -1]  # 0 is reachable set, -1 is target set
plot_2d(g, V_2D=V_2D)