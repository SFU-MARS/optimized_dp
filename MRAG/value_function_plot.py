import matplotlib.pyplot as plt
import numpy as np
from odp.Plots.plotting_utilities import *
from odp.Grid import Grid
from utilities import lo2slice1v1, lo2slice2v1, lo2slice1v0

# load reach-avoid value functions
value1v2 = np.load('MRAG/1v2AttackDefend_speed15.npy')
print(f'The shape of the 1v2 value function is {value1v2.shape} \n')

grid1v2 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30]))  # [30, 30, 30, 30, 30, 30][36, 36, 36, 36, 36, 36]


# 2v1 simulation
# initial positions
#attackers = [(0.0, 0.8), (0.2, 0.5)]  # [(-0.5, 0.0), (0.0, 0.8)], [(-0.5, 0.5), (-0.3, -0.8)], [(-0.5, -0.3), (0.8, -0.5)]
#defenders = [(-0.3, -0.3)] # [(0.3, -0.3)], [(0.0, 0.0)], [(0.3, 0.5)]
attackers = [(0.0, 0.0)] # [(-0.5, 0.0), (0.0, 0.8)]
defenders = [(-0.5, 0.0), (0.0, 0.8)]  #  [(-0.8, -0.3)]

ax = attackers[0][0]
ay = attackers[0][1]
d1x = defenders[0][0]
d1y = defenders[0][1]
d2x = defenders[1][0]
d2y = defenders[1][1]

# plot 1v2 reach-avoid set
jointstates2v1 = (ax, ay, d1x, d1y, d2x, d2y)
ax_slice, ay_slice, d1x_slice, d1y_slice, d2x_slice, d2y_slice = lo2slice2v1(jointstates2v1, slices=30)
# value_function2v1 = value2v1[:, :, a2x_slice, a2y_slice, d1x_slice, d1y_slice]
value_function1v2_1 = value1v2[ax_slice, ay_slice, :, :, d2x_slice, d2y_slice]
value_function1v2_2 = value1v2[ax_slice, ay_slice, d1x_slice, d1y_slice, :, :]
value_function1v2_3 = value1v2[:, :, d1x_slice, d1y_slice, d2x_slice, d2y_slice]

# plotting players
attackers_plot2 = [(ax, ay)]
defenders_plot2 = [(d1x, d1y), (d2x, d2y)]
# plot_game2v1_1(grid1v2, value_function2v1_1, attackers_plot2, defenders_plot2, name="$\mathcal{RA}^{21}_{\infty}$")
# plot_game2v1_2(grid1v2, value_function2v1_2, attackers_plot2, defenders_plot2, name="$\mathcal{RA}^{21}_{\infty}$")
plot_game1v2(grid1v2, value_function1v2_3, attackers_plot2, defenders_plot2, name="$\mathcal{RA}^{12}_{\infty}$")
# plot_game0(grid2v1, value_function2v1, attackers_plot2, defenders_plot2, name="$\mathcal{RA}^{21}_{\infty}$")
