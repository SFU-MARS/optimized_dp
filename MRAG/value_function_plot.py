import matplotlib.pyplot as plt
import numpy as np
from odp.Plots.plotting_utilities import *
from odp.Grid import Grid
from utilities import lo2slice1v1, lo2slice2v1, lo2slice1v0

# load reach-avoid value functions
grid_size1v1 = 45
grid_size1v2 = 35

value1v1 = np.load('MRAG/1v1AttackDefend_g45_dspeed1.5.npy')
# value1v1 = np.load('MRAG/1v1AttackDefend_g35_dspeed1.0.npy')
# value1v2 = np.load('MRAG/1v2AttackDefend_g35_dspeed1.5.npy')
# value1v2 = np.load('MRAG/1v2AttackDefend_g35_dspeed1.0.npy')
value1v2 = np.load('MRAG/1v2AttackDefend_g30_dspeed1.5.npy')


print(f"The shape of the 1v1 value function is {value1v1.shape} \n")
print(f'The shape of the 1v2 value function is {value1v2.shape} \n')

grid1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, 
               np.array([grid_size1v1, grid_size1v1, grid_size1v1, grid_size1v1]))  # original 45
grid1v2 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, 
               np.array([grid_size1v2, grid_size1v2, grid_size1v2, grid_size1v2, grid_size1v2, grid_size1v2]))  # [30, 30, 30, 30, 30, 30][36, 36, 36, 36, 36, 36]
# grid2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30]))  # [30, 30, 30, 30, 30, 30][36, 36, 36, 36, 36, 36]

# 2 vs. 1 simulation
# initial positions
# attackers_initials = [(0.0, 0.8), (0.2, 0.5)]  # [(-0.5, 0.0), (0.0, 0.8)], [(-0.5, 0.5), (-0.3, -0.8)], [(-0.5, -0.3), (0.8, -0.5)]
# defenders_initials = [(-0.3, -0.3)] # [(0.3, -0.3)], [(0.0, 0.0)], [(0.3, 0.5)]

# # 1 vs. 2 simulation
# initial positions
# attackers_initials = [(0.0, 0.0)] # 
# defenders_initials = [(-0.5, 0.4), (-0.5, -0.3)]  

# attackers_initials = [(-0.4, 0.0)] # 
# defenders_initials = [(-0.5, 0.5), (-0.5, -0.6)] 

attackers_initials =[(-0.1, 0.0)]  
defenders_initials = [(-0.5, 0.5), (-0.5, -0.1)] 

ax = attackers_initials[0][0]
ay = attackers_initials[0][1]
d1x = defenders_initials[0][0]
d1y = defenders_initials[0][1]
d2x = defenders_initials[1][0]
d2y = defenders_initials[1][1]

# plot 1v1 reach-avoid tube
jointstates1v1_1 = (ax, ay, d1x, d1y)
jointstates1v1_2 = (ax, ay, d2x, d2y)
ax_slice_1v1_1, ay_slice_1v1_1, d1x_slice_1v1_1, d1y_slice_1v1_1 = lo2slice1v1(jointstates1v1_1, slices=grid_size1v1)
ax_slice_1v1_2, ay_slice_1v1_2, d2x_slice_1v1_2, d2y_slice_1v1_2 = lo2slice1v1(jointstates1v1_2, slices=grid_size1v1)
value_function1v1_1 = value1v1[:, :, d1x_slice_1v1_1, d1y_slice_1v1_1]
# print(f"The initial value function between the attacker and the defender 1 is {value1v1[ax_slice_1v1_1, ay_slice_1v1_1, d1x_slice_1v1_1, d1y_slice_1v1_1]}. \n")
value_function1v1_2 = value1v1[:, :, d2x_slice_1v1_2, d2y_slice_1v1_2]
# print(f"The initial value function between the attacker and the defender 2 is {value1v1[ax_slice_1v1_2, ay_slice_1v1_2, d2x_slice_1v1_2, d2y_slice_1v1_2]}. \n")

# plot 1v2 reach-avoid tube
jointstates2v1 = (ax, ay, d1x, d1y, d2x, d2y)
ax_slice, ay_slice, d1x_slice, d1y_slice, d2x_slice, d2y_slice = lo2slice2v1(jointstates2v1, slices=grid_size1v2)
# value_function2v1 = value2v1[:, :, a2x_slice, a2y_slice, d1x_slice, d1y_slice]
# value_function1v2_1 = value1v2[ax_slice, ay_slice, :, :, d2x_slice, d2y_slice]
# value_function1v2_2 = value1v2[ax_slice, ay_slice, d1x_slice, d1y_slice, :, :]
value_function1v2_3 = value1v2[:, :, d1x_slice, d1y_slice, d2x_slice, d2y_slice]

# plotting players
attackers_plot2 = [(ax, ay)]
defenders_plot2 = [(d1x, d1y), (d2x, d2y)]
defenders_1v1_1 = [(d1x, d1y)]
defenders_1v1_2 = [(d2x, d2y)]
# plot_game2v1_1(grid1v2, value_function2v1_1, attackers_plot2, defenders_plot2, name="$\mathcal{RA}^{21}_{\infty}$")
# plot_game2v1_2(grid1v2, value_function2v1_2, attackers_plot2, defenders_plot2, name="$\mathcal{RA}^{21}_{\infty}$")
plot_game1v2(grid1v2, value_function1v2_3, attackers_plot2, defenders_plot2, name="$\mathcal{RA}^{12}_{\infty}$")
# plot_game0(grid2v1, value_function2v1, attackers_plot2, defenders_plot2, name="$\mathcal{RA}^{21}_{\infty}$")
# plot_game1v1(grid1v1, value_function1v1_1, attackers_plot2, defenders_1v1_1, name="$\mathcal{RA}^{11}_{\infty}$")
# plot_game1v1(grid1v1, value_function1v1_2, attackers_plot2, defenders_1v1_2, name="$\mathcal{RA}^{11}_{\infty}$")
