import math
import numpy as np
from Grid.GridProcessing import Grid
from dynamics.DubinsCar import DubinsCar
from spatialDerivatives.first_order_generic import spa_deriv
from Shapes.ShapesFunctions import CylinderShape
from solver import HJSolver
from plot_options import PlotOptions
from computeGraphs.graph_3D import spa_derivX, spa_derivY, spa_derivT
import heterocl as hcl


def heterocl_spa_deriv_3d(index, V, g):
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    i, j, k = index

    def hcl_spa_deriv_3d_schedule(hcl_deriv, hcl_V):
        with hcl.Stage("Calculate Derivative"):
            # hcl statements only work with hcl tensors
            hcl_i = hcl.scalar(i, name="i")
            hcl_j = hcl.scalar(j, name="j")
            hcl_k = hcl.scalar(k, name="k")

            dV_dx_L = hcl.scalar(0.0, "dV_dx_L")
            dV_dx_R = hcl.scalar(0.0, "dV_dx_R")

            dV_dy_L = hcl.scalar(0.0, "dV_dy_L")
            dV_dy_R = hcl.scalar(0.0, "dV_dy_R")

            dV_dT_L = hcl.scalar(0.0, "dV_dT_L")
            dV_dT_R = hcl.scalar(0.0, "dV_dT_R")

            dV_dx_L, dV_dx_R = spa_derivX(hcl_i, hcl_j, hcl_k, hcl_V, g)
            dV_dy_L, dV_dy_R = spa_derivY(hcl_i, hcl_j, hcl_k, hcl_V, g)
            dV_dT_L, dV_dT_R = spa_derivT(hcl_i, hcl_j, hcl_k, hcl_V, g)

            hcl_deriv[0] = (dV_dx_L + dV_dx_R) / 2
            hcl_deriv[1] = (dV_dy_L + dV_dy_R) / 2
            hcl_deriv[2] = (dV_dT_L + dV_dT_R) / 2

    hcl_deriv_ph = hcl.placeholder((3,), "deriv")
    hcl_V_ph = hcl.placeholder(V.shape, "V")
    s = hcl.create_schedule([hcl_deriv_ph, hcl_V_ph], hcl_spa_deriv_3d_schedule)
    hcl_spa_deriv_3d = hcl.build(s)

    hcl_spa_deriv = hcl.asarray(np.empty(3))
    hcl_V = hcl.asarray(V)
    hcl_spa_deriv_3d(hcl_spa_deriv, hcl_V)

    return hcl_spa_deriv.asnumpy()


def non_hetero_spa_deriv_3d(index, V, g):
    def spa_derivX(i, j, k):
        left_deriv = 0.0
        right_deriv = 0.0
        if i == 0:
            left_boundary = 0
            left_boundary = V[i, j, k] + np.abs(V[i + 1, j, k] - V[i, j, k]) * np.sign(V[i, j, k])
            left_deriv = (V[i, j, k] - left_boundary) / g.dx[0]
            right_deriv = (V[i + 1, j, k] - V[i, j, k]) / g.dx[0]
        elif i == V.shape[0] - 1:
            right_boundary = 0
            right_boundary = V[i, j, k] + np.abs(V[i, j, k] - V[i - 1, j, k]) * np.sign(V[i, j, k])
            left_deriv = (V[i, j, k] - V[i - 1, j, k]) / g.dx[0]
            right_deriv = (right_boundary - V[i, j, k]) / g.dx[0]
        elif i != 0 and i != V.shape[0] - 1:
            left_deriv = (V[i, j, k] - V[i - 1, j, k]) / g.dx[0]
            right_deriv = (V[i + 1, j, k] - V[i, j, k]) / g.dx[0]
        return left_deriv, right_deriv

    def spa_derivY(i, j, k):
        left_deriv = 0
        right_deriv = 0
        if j == 0:
            left_boundary = 0
            left_boundary = V[i, j, k] + np.abs(V[i, j + 1, k] - V[i, j, k]) * np.sign(V[i, j, k])
            left_deriv = (V[i, j, k] - left_boundary) / g.dx[1]
            right_deriv = (V[i, j + 1, k] - V[i, j, k]) / g.dx[1]
        elif j == V.shape[1] - 1:
            right_boundary = 0
            right_boundary = V[i, j, k] + np.abs(V[i, j, k] - V[i, j - 1, k]) * np.sign(V[i, j, k])
            left_deriv = (V[i, j, k] - V[i, j - 1, k]) / g.dx[1]
            right_deriv = (right_boundary - V[i, j, k]) / g.dx[1]
        elif j != 0 and j != V.shape[1] - 1:
            left_deriv = (V[i, j, k] - V[i, j - 1, k]) / g.dx[1]
            right_deriv = (V[i, j + 1, k] - V[i, j, k]) / g.dx[1]
        return left_deriv, right_deriv

    def spa_derivT(i, j, k):
        left_deriv = 0
        right_deriv = 0
        if k == 0:
            left_boundary = 0
            # left_boundary = V[i,j,50]
            left_boundary = V[i, j, V.shape[2] - 1]
            left_deriv = (V[i, j, k] - left_boundary) / g.dx[2]
            right_deriv = (V[i, j, k + 1] - V[i, j, k]) / g.dx[2]
        elif k == V.shape[2] - 1:
            right_boundary = 0
            right_boundary = V[i, j, 0]
            left_deriv = (V[i, j, k] - V[i, j, k - 1]) / g.dx[2]
            right_deriv = (right_boundary - V[i, j, k]) / g.dx[2]
        elif k != 0 and k != V.shape[2] - 1:
            left_deriv = (V[i, j, k] - V[i, j, k - 1]) / g.dx[2]
            right_deriv = (V[i, j, k + 1] - V[i, j, k]) / g.dx[2]
        return left_deriv, right_deriv

    i, j, k = index
    left_x, right_x = spa_derivX(i, j, k)
    left_y, right_y = spa_derivY(i, j, k)
    left_t, right_t = spa_derivT(i, j, k)

    return [(left_x + right_x) / 2, (left_y + right_y) / 2, (left_t + right_t) / 2]


def main():
    grid = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([40, 40, 40]), [2])
    dyn_sys = DubinsCar()
    ivf = CylinderShape(grid, [2], np.zeros(3), 1)

    lookback_length = 2.0
    t_step = 0.05
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
    V_all_t = HJSolver(dyn_sys, grid, ivf, tau, "minVWithV0", PlotOptions("3d_plot", [0, 1, 2], []),
                       save_all_t=True)

    test_dyn_sys = DubinsCar(x=np.array([2, 2, -np.pi]))
    index = grid.get_index(test_dyn_sys.x)
    derivs = spa_deriv(index, V_all_t[..., -1], grid, periodic_dims=[2])
    print(derivs)

    derivs2 = non_hetero_spa_deriv_3d(index, V_all_t[..., -1], grid)
    print(derivs2)

    derivs3 = heterocl_spa_deriv_3d(index, V_all_t[..., -1], grid)
    print(derivs3)


if __name__ in "__main__":
    main()
