import sys
sys.path.append("/Users/anjianli/Desktop/robotics/project/optimized_dp")

import numpy as np
from Grid.GridProcessing import grid
from Shapes.ShapesFunctions import *

import math

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class PlotReldyn5D(object):

    def plot(self):
        g = grid(np.array([-10.0, -10.0, -math.pi, 0, 0]), np.array([10.0, 10.0, math.pi, 17, 17]), 5,
                 np.array([41, 41, 31, 35, 35]), [2])

        # Local
        # V_1 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0729-wh_-0.1_0.1-ah_-5_-2-t5/reldyn5d_brs_wh_-0.1_0.1-ah_-5_-2_t_5.00.npy")
        # V_2 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0729-wh_full-ah_-2_1-t5/reldyn5d_brs_wh_full-ah_-2_1_t_5.00.npy")
        # V_3 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0729-wh_-0.2_0.2-ah_1_3-t5/reldyn5d_brs_wh_-0.2_0.2-ah_1_3_t_5.00.npy")
        # V_4 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0725-full_range-t5/reldyn5d_brs_t_5.00.npy")

        # Remote
        V_1 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0730-wh_-0.1_0.1-ah_-5_-2-t5/reldyn5d_brs_wh_-0.1_0.1-ah_-5_-2_t_2.00.npy")
        V_2 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0730-wh_full-ah_-2_1-t5/reldyn5d_brs_wh_full-ah_-2_1_t_2.00.npy")
        V_3 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0730-wh_-0.2_0.2-ah_1_3-t5/reldyn5d_brs_wh_-0.2_0.2-ah_1_3_t_2.00.npy")
        V_4 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0730-full_range-t5/reldyn5d_brs_full_range_t_2.00.npy")

        x_grid, y_grid = self.get_xy_grid(g, [0, 1])

        # Slice the value function
        # Psi_relative, v_human, v_robot
        # val_1 = np.squeeze(V_1[:, :, 19, 34, 0])
        # val_2 = np.squeeze(V_2[:, :, 19, 34, 0])
        # val_3 = np.squeeze(V_3[:, :, 19, 34, 0])
        # val_4 = np.squeeze(V_4[:, :, 19, 34, 0])

        val_1 = np.squeeze(V_1[:, :, 23, 24, 10])
        val_2 = np.squeeze(V_2[:, :, 23, 24, 10])
        val_3 = np.squeeze(V_3[:, :, 23, 24, 10])
        val_4 = np.squeeze(V_4[:, :, 23, 24, 10])

        fig, ax = plt.subplots()

        CS_1 = ax.contour(x_grid, y_grid, val_1, levels=[0], colors='darkmagenta')
        ax.clabel(CS_1, inline=1, fontsize=5)
        CS_2 = ax.contour(x_grid, y_grid, val_2, levels=[0], colors='limegreen')
        ax.clabel(CS_2, inline=1, fontsize=5)
        CS_3 = ax.contour(x_grid, y_grid, val_3, levels=[0], colors='steelblue')
        ax.clabel(CS_3, inline=1, fontsize=5)
        CS_4 = ax.contour(x_grid, y_grid, val_4, levels=[0], colors='gold')
        ax.clabel(CS_4, inline=1, fontsize=5)

        lines = [CS_1.collections[0], CS_2.collections[0], CS_3.collections[0], CS_4.collections[0]]
        labels = ['Mode1: wh_-0.1_0.1-ah_-5_-2', 'Mode2: wh_full-ah_-2_1', 'Mode3: wh_-0.2_0.2-ah_1_3', 'Mode0: full_range']

        ax.legend(lines, labels)

        ax.set_xlabel("x_relative")
        ax.set_ylabel("y_relative")
        ax.set_title('Avoid set: psi_rel = pi/2, v_human = 12m/s, v_robot = 5m/s')

        plt.show()

    def get_xy_grid(self, grid, dims_plot):

        if len(dims_plot) != 2:
            raise Exception('dims_plot length should be equal to 2\n')
        else:
            dim1, dim2 = dims_plot[0], dims_plot[1]
            complex_x = complex(0, grid.pts_each_dim[dim1])
            complex_y = complex(0, grid.pts_each_dim[dim2])
            mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x,
                               grid.min[dim2]:grid.max[dim2]: complex_y]

        return mg_X, mg_Y


if __name__ == '__main__':
    PlotReldyn5D().plot()