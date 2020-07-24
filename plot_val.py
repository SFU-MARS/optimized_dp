import numpy as np
from Grid.GridProcessing import grid
from Shapes.ShapesFunctions import *

# Specify the  file that includes dynamic systems
from Plots.plotting_utilities import *

import math

def main():
    g = grid(np.array([-10.0, -10.0, -math.pi, 0, 0]), np.array([10.0, 10.0, math.pi, 17, 17]), 5,
             np.array([41, 41, 31, 35, 35]), [2])

    V_1 = np.load("/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/reldyn5d_brs_t_02.npy")

    # x_rel, y_rel, psi_rel, v_human, v_robot
    # print(np.min(V_1[:, :, :, 0, 0]))
    # print(np.min(V_1[:, :, :, 34, 0]))
    print(np.min(V_1[:, :, :, 34, 17]))


    plot_isosurface(g, V_1, [0, 1, 2])

if __name__ == '__main__':
    main()