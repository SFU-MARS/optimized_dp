import matplotlib.pyplot as plt
import numpy as np
from odp.Plots.plotting_utilities import *
from odp.Grid import Grid
from utilities import lo2slice1v1, lo2slice2v1, lo2slice1v0
from odp.solver import HJSolver, computeSpatDerivArray

# load reach-avoid sets 
# value1v1 = np.load('MRAG/1v1AttackDefend.npy')
value1v1 = np.load('MRAG/1v1AttackDefend_speed15.npy')  # grid = 45
# value2v1 = np.load('MRAG/2v1AttackDefend.npy')
# value2v1 = np.load('2v1AttackDefend_subset.npy')  # the subset one
value2v1 = np.load('2v1AttackDefend_speed15.npy')  # the one we use in paper
grids1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))
grid2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30]))  # [30, 30, 30, 30, 30, 30][36, 36, 36, 36, 36, 36]
print(f'The shape of the 1v1 value function is {value1v1.shape} \n')
print(f'The shape of the 2v1 value function is {value2v1.shape} \n')


# 2v1 simulation
# initial positions
#attackers = [(0.0, 0.8), (0.2, 0.5)]  # [(-0.5, 0.0), (0.0, 0.8)], [(-0.5, 0.5), (-0.3, -0.8)], [(-0.5, -0.3), (0.8, -0.5)]
#defenders = [(-0.3, -0.3)] # [(0.3, -0.3)], [(0.0, 0.0)], [(0.3, 0.5)]
attackers = [(-0.5, 0.0), (0.0, 0.8)] # [(-0.5, 0.0), (0.0, 0.8)]
defenders = [(0.3, -0.3)]  #  [(-0.8, -0.3)]

a1x = attackers[0][0]
a1y = attackers[0][1]
a2x = attackers[1][0]
a2y = attackers[1][1]
d1x = defenders[0][0]
d1y = defenders[0][1]

# plot 1v1 reach-avoid set 
jointstates1v1_1 = (a1x, a1y, d1x, d1y)
jointstates1v1_2 = (a2x, a2y, d1x, d1y)
a1x_slice, a1y_slice, d1x_slice, d1y_slice = lo2slice1v1(jointstates1v1_1, slices=45)
a2x_slice, a2y_slice, d1x_slice, d1y_slice = lo2slice1v1(jointstates1v1_2, slices=45)
value_function1v1 = value1v1[:, :, d1x_slice, d1y_slice]  # , 0] if the saveAllTimeSteps=True. 0 is reachable set, -1 is target set
# plotting players
attackers_plot1 = [(a1x, a1y), (a2x, a2y)]  # (a2x, a2y)
defenders_plot1 = [(d1x, d1y)]
plot_game1v1(grids1v1, value_function1v1, attackers_plot1, defenders_plot1, name="$\mathcal{RA}^{11}_{\infty}$")
# plot_game0(grids1v1, value_function1v1, attackers_plot1, defenders_plot1, name="$\mathcal{RA}^{21}_{\infty}$")

# plot 2v1 reach-avoid set
jointstates2v1 = (a1x, a1y, a2x, a2y, d1x, d1y)
a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice = lo2slice2v1(jointstates2v1, slices=30)
# value_function2v1 = value2v1[:, :, a2x_slice, a2y_slice, d1x_slice, d1y_slice]
value_function2v1_1 = value2v1[a1x_slice, a1y_slice, :, :, d1x_slice, d1y_slice]
value_function2v1_2 = value2v1[a1x_slice, a1y_slice, a2x_slice, a2y_slice, :, :]

# plotting players
attackers_plot2 = [(a1x, a1y), (a2x, a2y)]
defenders_plot2 = [(d1x, d1y)]
plot_game2v1_1(grid2v1, value_function2v1_1, attackers_plot2, defenders_plot2, name="$\mathcal{RA}^{21}_{\infty}$")
plot_game2v1_2(grid2v1, value_function2v1_2, attackers_plot2, defenders_plot2, name="$\mathcal{RA}^{21}_{\infty}$")
# plot_game0(grid2v1, value_function2v1, attackers_plot2, defenders_plot2, name="$\mathcal{RA}^{21}_{\infty}$")



# # plot 1v1 reach-avoid game set
# attackers = [(0.0, 0.0), (0.0, 0.8), (-0.5, 0.0), (0.5, -0.5)]
# defenders = [(0.3, 0.5), (-0.3, -0.5)] # 
# grids1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))
# value1v1 = np.load('MRAG/1v1AttackDefend.npy')
# # value1v1 = value1v1[..., 0]
# print(f'The shape of the value function is {value1v1.shape} \n')
# # define the joint states of (a1x, a1y, d1x, d1y)
# a1x = 0
# a1y = 0
# d1x = defenders[0][0]
# d1y = defenders[0][1]
# jointstates1v1 = (a1x, a1y, d1x, d1y)
# a1x_slice, a1y_slice, d1x_slice, d1y_slice = lo2slice1v1(jointstates1v1, slices=45)
# print(f'The attacker is at the location [{a1x}, {a1y}] and the defender is at the location [{d1x}, {d1y}] \n')
# print(f'The value function of the attacker at the location (0, 0) is {value1v1[a1x_slice, a1y_slice, d1x_slice, d1y_slice]}. \n')
# value_function1v1 = value1v1[:, :, d1x_slice, d1y_slice]  # , 0] if the saveAllTimeSteps=True. 0 is reachable set, -1 is target set
# print(f'The shape of the 1v1 value function is {value_function1v1.shape}. \n')
# # plot_2d(grids1v1, value_function1v1)
# # if want to add the positons of attackers and defenders
# attackers_plot = [(0.0, 0.0), (0.0, 0.8), (-0.5, 0.0), (0.5, -0.5)]
# defenders_plot = [(d1x, d1y)]
# plot_game(grids1v1, value_function1v1, attackers_plot, defenders_plot)
# plot_game0(grids1v1, value_function1v1, attackers_plot, defenders_plot)

# # plot for 2v1 game set
# grid2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
#                6, np.array([36, 36, 36, 36, 36, 36])) # original 45, on mars-14 20 is the upper bound
# value2v1 = np.load('MRAG/2v1AttackDefend.npy')
# print(f'The shape of the value function is {value2v1.shape} \n')
# # define the joint states of (a1x, a1y, a2x, a2y, d1x, d1y)
# attackers = [(0.0, 0.0), (0.0, 0.8)]  # [(0.0, 0.0), (0.0, 0.8), (-0.5, 0.0), (0.5, -0.5), (-0.5, -0.3), (0.8, -0.5)]
# defenders = [(0.3, 0.5)] # [(0.3, 0.5), (-0.3, -0.5)]
# a1x = attackers[0][0]
# a1y = attackers[0][1]
# a2x = attackers[1][0]
# a2y = attackers[1][1]
# d1x = defenders[0][0]
# d1y = defenders[0][1]
# jointstates2v1 = (a1x, a1y, a2x, a2y, d1x, d1y)
# # attackers = [(a1x, a1y), (a2x, a2y)]
# a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice = lo2slice2v1(jointstates2v1, slices=30)
# #
# value_function2v1 = value2v1[:, :, a2x_slice, a2y_slice, d1x_slice, d1y_slice]
# print("Min value of the array {}".format(np.min(value_function2v1)))
# print(f'The shape of the 2v1 value function is {value_function2v1.shape}. \n')
# print(f'The HJ value of the current position {(a1x, a1y, a2x, a2y, d1x, d1y)} is {value2v1[a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice]}. \n')
# attackers_plot = [(a2x, a2y)]
# defenders_plot = [(d1x, d1y)]
# plot_game(grid2v1, value_function2v1, attackers_plot, defenders_plot)
# plot_game0(grid2v1, value_function2v1, attackers_plot, defenders_plot)


# # plot 1v0 reach-avoid game
# grids1v0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) # original 45
# value1v0 = np.load('MRAG/1v0AttackDefend.npy')
# print(f'The shape of the value function is {value1v0.shape} \n')
# # define the joint states of (a1x, a1y, d1x, d1y)
# a1x = 0
# a1y = 0
# jointstates1v0 = (a1x, a1y)
# a1x_slice, a1y_slice = lo2slice1v0(jointstates1v0, slices=100)
# # print(f'The attacker is at the location [{a1x}, {a1y}] and the defender is at the location [{d1x}, {d1y}] \n')
# print(f'The value function of the attacker at the location (0, 0) is {value1v0[a1x_slice, a1y_slice]}. \n')
# value_function1v0 = value1v0[:,:,0]  # , 0] if the saveAllTimeSteps=True. 0 is reachable set, -1 is target set
# print(f'The shape of the 1v1 value function is {value_function1v0.shape}. \n')
# # plot_2d(grids1v1, value_function1v1)
# # if want to add the positons of attackers and defenders
# attackers = [(a1x, a1y)]
# # plot_game(grids1v0, value_function1v0, attackers, defenders)
# plot_original(grids1v0, value_function1v0)

# 1 vs. 2 value function check by plotting
