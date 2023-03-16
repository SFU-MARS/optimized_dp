import matplotlib.pyplot as plt
import numpy as np
from odp.Plots.plotting_utilities import plot_2d, plot_game
from odp.Grid import Grid
from utilities import lo2slice1v1, lo2slice2v1
from odp.solver import HJSolver, computeSpatDerivArray
import math

# plot 1v1 reach-avoid game
grids1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))
value1v1 = np.load('MRAG/1v1AttackDefend.npy')
print(f'The shape of the value function is {value1v1.shape} \n')
# define the joint states of (a1x, a1y, d1x, d1y)
a1x = 0
a1y = 0
d1x = -0.3
d1y = 0.5
jointstates1v1 = (a1x, a1y, d1x, d1y)
a1x_slice, a1y_slice, d1x_slice, d1y_slice = lo2slice1v1(jointstates1v1, slices=45)
print(f'The attacker is at the location [{a1x}, {a1y}] and the defender is at the location [{d1x}, {d1y}] \n')
print(f'The value function of the attacker at the location (0, 0) is {value1v1[a1x_slice, a1y_slice, d1x_slice, d1y_slice]}. \n')
value_function1v1 = value1v1[:, :, d1x_slice, d1y_slice]  # , 0] if the saveAllTimeSteps=True. 0 is reachable set, -1 is target set
print(f'The shape of the 1v1 value function is {value_function1v1.shape}. \n')
# plot_2d(grids1v1, value_function1v1)
# if want to add the positons of attackers and defenders
attackers = [(a1x, a1y)]
defenders = [(d1x, d1y)]
plot_game(grids1v1, value_function1v1, attackers, defenders)

# plot for 2v1 game
grid2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30])) # original 45, on mars-14 20 is the upper bound
value2v1 = np.load('MRAG/2v1AttackDefend.npy')
print(f'The shape of the value function is {value2v1.shape} \n')
# define the joint states of (a1x, a1y, a2x, a2y, d1x, d1y)
a1x = 0
a1y = 0
a2x = 0.7
a2y = 0.2
d1x = -0.3
d1y = 0.5
jointstates2v1 = (a1x, a1y, a2x, a2y, d1x, d1y)
attackers = [(a1x, a1y), (a2x, a2y)]
defenders = [(d1x, d1y)]
a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice = lo2slice2v1(jointstates2v1, slices=30)
# 
value_function2v1 = value2v1[:, :, a2x_slice, a2y_slice, d1x_slice, d1y_slice] 
print("Min value of the array {}".format(np.min(value_function2v1)))
print(f'The shape of the 2v1 value function is {value_function2v1.shape}. \n')
plot_2d(grid2v1, value_function2v1)
plot_game(grid2v1, value_function2v1, attackers, defenders)