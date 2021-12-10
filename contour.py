import numpy as np
def plot_contour(grid, V_all_t):
    """
    Args:
        grid:
        V_all_t:
    Returns:
    """

    v_min, v_max = V_all_t.min(), V_all_t.max()
    levels = np.linspace(round(v_min), round(v_max), round(v_max) - round(v_min) + 1)

    dim1, dim2, dim3 = (0, 1, 2)
    complex_x = complex(0, grid.pts_each_dim[dim1])
    complex_y = complex(0, grid.pts_each_dim[dim2])
    mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]:complex_y]

    # fig = plt.figure(figsize=(13, 8))
    fig = plt.figure()
    ax = plt.axes()
    plt.jet()

    V_all_t = (V_all_t >= 0)
    plt.contourf(mg_X, mg_Y, V_all_t[:, :, 59], vmin=v_min, vmax=v_max, levels=levels)

    plt.show()


V = np.load("V_r1.15_grid101.npy")
from Grid.GridProcessing import Grid
import math
import matplotlib.pyplot as plt
g = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([101, 101, 101]), [2])
print(g.get_index((-3, -3, 0)))
plot_contour(g, V)