import numpy as np
from Grid.GridProcessing import grid
from Shapes.ShapesFunctions import *

# Specify the  file that includes dynamic systems
from Plots.plotting_utilities import *

import math

def main():
    g = grid(np.array([-10.0, -10.0, -math.pi, 0, 0]), np.array([10.0, 10.0, math.pi, 17, 17]), 5,
             np.array([41, 41, 31, 35, 35]), [2])

    # Local
    V_1 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0729-wh_-0.1_0.1-ah_-5_-2-t5/reldyn5d_brs_wh_-0.1_0.1-ah_-5_-2_t_2.00.npy")
    V_2 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0729-wh_full-ah_-2_1-t5/reldyn5d_brs_wh_full-ah_-2_1_t_2.00.npy")
    V_3 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0729-wh_-0.2_0.2-ah_1_3-t5/reldyn5d_brs_wh_-0.2_0.2-ah_1_3_t_2.00.npy")
    V_4 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0725-full_range-t5/reldyn5d_brs_t_2.00.npy")

    # Remote
    # V_1 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0725-full_range-t5/reldyn5d_brs_t_5.00.npy")
    # V_2 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0725-full_range-t5/reldyn5d_brs_t_4.50.npy")

    # index1 = np.zeros(shape=np.shape(V_1))
    # index2 = np.zeros(shape=np.shape(V_2))

    # for i in range(np.shape(V_1)[0]):
    #     print(i)
    #     for j in range(np.shape(V_1)[1]):
    #         for k in range(np.shape(V_1)[2]):
    #             for l in range(np.shape(V_1)[3]):
    #                 for m in range(np.shape(V_1)[4]):
    #                     if V_1[i][j][k][l][m] <= 0:
    #                         index1[i][j][k][l][m] = 1
    #                     if V_2[i][j][k][l][m] <= 0:
    #                         index2[i][j][k][l][m] = 1
    #
    #
    # print(np.sum(index1))
    # print(np.sum(index2))


    # print(np.max(np.abs(V_2 - V_1)))

    # x_rel, y_rel, psi_rel, v_human, v_robot
    # print(np.min(V_1[:, :, :, 0, 0]))
    # print(np.min(V_1[:, :, :, 34, 0]))
    # print(np.min(V_1[:, :, :, 34, 17]))

    # plot_isosurface(g, V_1, [0, 1, 2])
    # plot_isosurface(g, V_2, [0, 1, 2])
    plot_isosurface(g, V_3, [0, 1, 2])
    # plot_isosurface(g, V_4, [0, 1, 2])

if __name__ == '__main__':
    main()