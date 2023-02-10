import matplotlib.pyplot as plt
import numpy as np
from odp.Plots.plotting_utilities import plot_2d, plot_2d_with_avoid
from odp.Grid import Grid
from utilities import loca2slices, check
from odp.solver import HJSolver, computeSpatDerivArray
import math

# plot value_function in 2d figure
# g must be the same as one in the ValueFunction.py
g = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))
value_function = np.load('1v1AttackDefend.npy')
print(f'The shape of the value function is {value_function.shape} \n')
# define the locations of the defender
x_d = -0.3
y_d = -0.5
x_defender, y_defender = loca2slices(x_location=x_d, y_location=y_d, slices=45)
print(f'The defender is at the location [{x_d}, {y_d}] \n')
V_2D = value_function[:, :, x_defender, y_defender, 0]  # 0 is reachable set, -1 is target set
plot_2d(g, V_2D=V_2D)

# plot value_function in 2d figure for debug
# g must be the same as one in the ValueFunction.py
# g_debug = Grid(np.array([-1.0, -1.0, -math.pi]), np.array([1.0, 1.0, math.pi]), 3, np.array([40, 40, 40]), [2])
# value_function = np.load('example_reach_avoid.npy')
# print(f'The shape of the value function is {value_function.shape} \n')
# # define the locations of the defender
# # x_d = -0.3
# # y_d = 0.5
# # x_defender, y_defender = loca2slices(x_location=x_d, y_location=y_d, slices=45)
# # print(f'The defender is at the location [{x_d}, {y_d}] \n')
# V_2D = value_function[:, :, 20, -1]  # 0 is reachable set, -1 is target set
# plot_2d(g_debug, V_2D=V_2D)


# plot the 2D map
# plt.xlim((-1.0, 1.0))
# plt.ylim((-1.0, 1.0))
# plt.xlabel('x-label')
# plt.ylabel('y-label')
# # obs1
# plt.hlines(y=-0.3, xmin=-0.1, xmax=0.1, colors='blue', linewidth=2.0)
# plt.hlines(y=-1.0, xmin=-0.1, xmax=0.1, colors='blue', linewidth=2.0)
# plt.vlines(x=-0.1, ymin=-1.0, ymax=-0.3, colors='blue', linewidth=2.0)
# plt.vlines(x=0.1, ymin=-1.0, ymax=-0.3, colors='blue', linewidth=2.0)
# # obs2
# plt.hlines(y=0.6, xmin=-0.1, xmax=0.1, colors='blue', linewidth=2.0)
# plt.hlines(y=0.3, xmin=-0.1, xmax=0.1, colors='blue', linewidth=2.0)
# plt.vlines(x=-0.1, ymin=0.3, ymax=0.6, colors='blue', linewidth=2.0)
# plt.vlines(x=0.1, ymin=0.3, ymax=0.6, colors='blue', linewidth=2.0)
# # target
# plt.hlines(y=0.1, xmin=0.6, xmax=0.8, colors='green', linewidth=2.0)
# plt.hlines(y=0.3, xmin=0.6, xmax=0.8, colors='green', linewidth=2.0)
# plt.vlines(x=0.6, ymin=0.1, ymax=0.3, colors='green', linewidth=2.0)
# plt.vlines(x=0.8, ymin=0.1, ymax=0.3, colors='green', linewidth=2.0)
# # legends
# plt.text(-0.05, -0.60, 'obs1')
# plt.text(-0.05, 0.42, 'obs2')
# plt.text(0.62, 0.18, 'target')
# plt.show()


