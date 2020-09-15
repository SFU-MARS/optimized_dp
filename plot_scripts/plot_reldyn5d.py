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
                 np.array([41, 41, 36, 35, 35]), [2])


        # TODO 0909
        ctrl_beta_0 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0910/mode-1/reldyn5d_ctrl_beta_mode-1_t_3.00.npy")
        ctrl_acc_0 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0910/mode-1/reldyn5d_ctrl_acc_mode-1_t_3.00.npy")

        # Set time
        time = 3.0
        # Set the slice parameter, not psi_rel = psi_h - psi_r
        # psi = 90
        # v_h = 6.0
        # v_r = 0.0

        psi = - 90
        v_h = 6.0
        v_r = 0.0

        V_0 = np.load(
            "/home/anjianl/Desktop/project/optimized_dp/data/brs/0910/mode0/reldyn5d_brs_mode0_t_{:.2f}.npy".format(time))
        V_1 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0910/mode1/reldyn5d_brs_mode1_t_{:.2f}.npy".format(time))
        V_2 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0910/mode2/reldyn5d_brs_mode2_t_{:.2f}.npy".format(time))
        V_3 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0910/mode3/reldyn5d_brs_mode3_t_{:.2f}.npy".format(time))
        V_4 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0910/mode4/reldyn5d_brs_mode4_t_{:.2f}.npy".format(time))
        V_5 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0910/mode5/reldyn5d_brs_mode5_t_{:.2f}.npy".format(time))
        V_6 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0910/mode-1/reldyn5d_brs_mode-1_t_{:.2f}.npy".format(time))

        x_grid, y_grid = self.get_xy_grid(g, [0, 1])

        psi_index = int((psi + 180) / 10)
        v_h_index = int(2 * v_h)
        v_r_index = int(2 * v_r)

        val_0 = np.squeeze(V_0[:, :, psi_index, v_h_index, v_r_index])
        val_1 = np.squeeze(V_1[:, :, psi_index, v_h_index, v_r_index])
        val_2 = np.squeeze(V_2[:, :, psi_index, v_h_index, v_r_index])
        val_3 = np.squeeze(V_3[:, :, psi_index, v_h_index, v_r_index])
        val_4 = np.squeeze(V_4[:, :, psi_index, v_h_index, v_r_index])
        val_5 = np.squeeze(V_5[:, :, psi_index, v_h_index, v_r_index])
        val_6 = np.squeeze(V_6[:, :, psi_index, v_h_index, v_r_index])

        ctrl_beta_val_0 = np.squeeze(ctrl_beta_0[:, :, psi_index, v_h_index, v_r_index])
        ctrl_acc_val_0 = np.squeeze(ctrl_acc_0[:, :, psi_index, v_h_index, v_r_index])

        fig, ax = plt.subplots()

        CS_0 = ax.contour(x_grid, y_grid, val_0, levels=[0], colors='sandybrown')
        ax.clabel(CS_0, inline=1, fontsize=5)
        CS_1 = ax.contour(x_grid, y_grid, val_1, levels=[0], colors='darkmagenta')
        ax.clabel(CS_1, inline=1, fontsize=5)
        CS_2 = ax.contour(x_grid, y_grid, val_2, levels=[0], colors='limegreen')
        ax.clabel(CS_2, inline=1, fontsize=5)
        CS_3 = ax.contour(x_grid, y_grid, val_3, levels=[0], colors='steelblue')
        ax.clabel(CS_3, inline=1, fontsize=5)
        CS_4 = ax.contour(x_grid, y_grid, val_4, levels=[0], colors='pink')
        ax.clabel(CS_4, inline=1, fontsize=5)
        CS_5 = ax.contour(x_grid, y_grid, val_5, levels=[0], colors='gold')
        ax.clabel(CS_5, inline=1, fontsize=5)
        CS_6 = ax.contour(x_grid, y_grid, val_6, levels=[0], colors='red')
        ax.clabel(CS_6, inline=1, fontsize=5)

        # lines = [CS_1.collections[0], CS_2.collections[0], CS_3.collections[0], CS_4.collections[0]]
        # labels = ['Mode 1: stable', 'Mode 2: acceleration', 'Mode 3: left turn', 'Mode 4: right turn']

        # lines = [CS_3.collections[0], CS_4.collections[0]]
        # labels = ['Mode 3: left turn', 'Mode 4: right turn']

        lines = [CS_0.collections[0], CS_1.collections[0], CS_2.collections[0], CS_3.collections[0], CS_4.collections[0], CS_5.collections[0], CS_6.collections[0]]
        labels = ['Mode 0: decelerate', 'Mode 1: stable', 'Mode2: acceleration', 'Mode 3: left turn', 'Mode 4: right turn', 'Mode 5: in roundabout', "Mode -1, full range"]

        ax.legend(lines, labels)

        ax.set_xlabel("x_relative")
        ax.set_ylabel("y_relative")
        ax.set_title('Avoid set: psi_rel = {:d}, v_human = {:.1f}m/s, v_robot = {:.1f}m/s, t = {:.1f}s'.format(psi, v_h, v_r, time))


        # Plot control
        # CS_0 = ax.contour(x_grid, y_grid, ctrl_acc_val_0, levels=[-5, 3], colors='brown')
        # ax.clabel(CS_0, inline=1, fontsize=5)
        # ax.set_title('control acc')
        # CS_0 = ax.contour(x_grid, y_grid, ctrl_beta_val_0, levels=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], colors='magenta')
        # ax.clabel(CS_0, inline=1, fontsize=5)
        # ax.set_title('control beta')


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