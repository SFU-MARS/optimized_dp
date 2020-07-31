import sys
sys.path.append("/Users/anjianli/Desktop/robotics/project/optimized_dp")

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
    V_1 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0730-full_range-t5/reldyn5d_brs_full_range_t_2.00.npy")
    # V_2 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0729-wh_full-ah_-2_1-t5/reldyn5d_brs_wh_full-ah_-2_1_t_5.00.npy")
    # V_3 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0729-wh_-0.2_0.2-ah_1_3-t5/reldyn5d_brs_wh_-0.2_0.2-ah_1_3_t_5.00.npy")
    # V_4 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0725-full_range-t5/reldyn5d_brs_t_5.00.npy")

    # V_5 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0729-wh_full-ah_-2_1-t5/reldyn5d_brs_wh_full-ah_-2_1_t_4.00.npy")

    # print(np.max(V_5 - V_2))
    # print(np.min(V_5 - V_2))

    # Remote
    # V_1 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0725-full_range-t5/reldyn5d_brs_t_5.00.npy")
    # V_2 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0725-full_range-t5/reldyn5d_brs_t_4.50.npy")

    plot_isosurface(g, V_1, [0, 1, 2])
    # plot_isosurface(g, V_2, [0, 1, 2])
    # plot_isosurface(g, V_3, [0, 1, 2])
    # plot_isosurface(g, V_4, [0, 1, 2])

if __name__ == '__main__':
    main()
