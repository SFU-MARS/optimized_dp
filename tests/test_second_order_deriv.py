from hashlib import shake_256
import heterocl as hcl
import odp
import numpy as np

from odp.spatialDerivatives.second_orderENO3D import (
    secondOrderT,
    secondOrderX,
    secondOrderY,
)
from odp.spatialDerivatives.second_orderENO4D import (
    secondOrderX1_4d,
    secondOrderX2_4d,
    secondOrderX3_4d,
    secondOrderX4_4d,
)
from odp.spatialDerivatives.second_orderENO5D import (
    secondOrderX1_5d,
    secondOrderX2_5d,
    secondOrderX3_5d,
    secondOrderX4_5d,
    secondOrderX5_5d,
)
from odp.spatialDerivatives.second_orderENO6D import (
    secondOrderX1_6d,
    secondOrderX2_6d,
    secondOrderX3_6d,
    secondOrderX4_6d,
    secondOrderX5_6d,
    secondOrderX6_6d,
)
from odp.Grid import Grid

hcl.init()


def test_heterocl_spa_deriv_3d():
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    grid = Grid(np.array([-5, -5, -5]), np.array([5, 5, 5]), 3, np.array([10, 10, 10]))
    X, Y, Z = np.meshgrid(*grid.grid_points, indexing="ij", sparse=True)

    def f(x, y, z):
        return x**2 + y**2 + z**2

    V = f(X, Y, Z)
    # pick points at grid points

    def hcl_spa_deriv_3d_schedule(hcl_deriv, hcl_V, hcl_index):
        with hcl.Stage("Calculate Derivative"):
            # hcl statements only work with hcl tensors
            hcl_i = hcl.scalar(hcl_index[0], name="i")
            hcl_j = hcl.scalar(hcl_index[1], name="j")
            hcl_k = hcl.scalar(hcl_index[2], name="k")

            dV_dx_L = hcl.scalar(0.0, "dV_dx_L")
            dV_dx_R = hcl.scalar(0.0, "dV_dx_R")

            dV_dy_L = hcl.scalar(0.0, "dV_dy_L")
            dV_dy_R = hcl.scalar(0.0, "dV_dy_R")

            dV_dT_L = hcl.scalar(0.0, "dV_dT_L")
            dV_dT_R = hcl.scalar(0.0, "dV_dT_R")

            dV_dx_L, dV_dx_R = secondOrderX(hcl_i, hcl_j, hcl_k, hcl_V, grid)
            dV_dy_L, dV_dy_R = secondOrderY(hcl_i, hcl_j, hcl_k, hcl_V, grid)
            dV_dT_L, dV_dT_R = secondOrderT(hcl_i, hcl_j, hcl_k, hcl_V, grid)

            hcl_deriv[0] = (dV_dx_L + dV_dx_R) / 2
            hcl_deriv[1] = (dV_dy_L + dV_dy_R) / 2
            hcl_deriv[2] = (dV_dT_L + dV_dT_R) / 2

    hcl_deriv_ph = hcl.placeholder((3,), "deriv")
    hcl_V_ph = hcl.placeholder(V.shape, "V")
    hcl_index_ph = hcl.placeholder((3,), "index")
    s = hcl.create_schedule(
        [hcl_deriv_ph, hcl_V_ph, hcl_index_ph], hcl_spa_deriv_3d_schedule
    )
    hcl_spa_deriv_3d = hcl.build(s)

    hcl_spa_deriv = hcl.asarray(np.empty(3))
    hcl_V = hcl.asarray(V)

    def compute_spa_deriv(state):
        hcl_index = hcl.asarray(grid.get_index(state))
        hcl_spa_deriv_3d(hcl_spa_deriv, hcl_V, hcl_index)
        return hcl_spa_deriv.asnumpy()

    state = np.array([grid.grid_points[i][3] for i in range(3)])
    # index = grid.get_index(state)
    spa_deriv = compute_spa_deriv(state)
    ans = 2 * state
    print(spa_deriv, ans)

    assert np.allclose(spa_deriv, ans)

    # state = np.array(
    #     [(grid.grid_points[i][3] + grid.grid_points[i][4]) / 2 for i in range(3)]
    # )
    # spa_deriv = compute_spa_deriv(state)
    # ans = 2 * state
    # print(spa_deriv, ans)

    # assert np.allclose(spa_deriv, ans)




def test_heterocl_spa_deriv_4d():
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    DIM = 4

    grid = Grid(
        np.array([-5, -5, -5, -5]),
        np.array([5, 5, 5, 5]),
        DIM,
        np.array([10, 10, 10, 10]),
    )
    X1, X2, X3, X4 = np.meshgrid(*grid.grid_points, indexing="ij", sparse=True)

    def f(x1, x2, x3, x4):
        return x1**2 + x2**2 + x3**2 + x4**2

    V = f(X1, X2, X3, X4)
    # pick points at grid points

    def hcl_spa_deriv_4d_schedule(hcl_deriv, hcl_V, hcl_index):
        with hcl.Stage("Calculate Derivative"):
            # hcl statements only work with hcl tensors
            hcl_i = hcl.scalar(hcl_index[0], name="i")
            hcl_j = hcl.scalar(hcl_index[1], name="j")
            hcl_k = hcl.scalar(hcl_index[2], name="k")
            hcl_l = hcl.scalar(hcl_index[3], name="l")

            dV_dx1_L = hcl.scalar(0.0, "dV_dx1_L")
            dV_dx1_R = hcl.scalar(0.0, "dV_dx1_R")

            dV_dx2_L = hcl.scalar(0.0, "dV_dx2_L")
            dV_dx2_R = hcl.scalar(0.0, "dV_dx2_R")

            dV_dx3_L = hcl.scalar(0.0, "dV_dx3_L")
            dV_dx3_R = hcl.scalar(0.0, "dV_dx3_R")

            dV_dx4_L = hcl.scalar(0.0, "dV_dx4_L")
            dV_dx4_R = hcl.scalar(0.0, "dV_dx4_R")

            dV_dx1_L, dV_dx1_R = secondOrderX1_4d(hcl_i, hcl_j, hcl_k, hcl_l, hcl_V, grid)
            dV_dx2_L, dV_dx2_R = secondOrderX2_4d(hcl_i, hcl_j, hcl_k, hcl_l, hcl_V, grid)
            dV_dx3_L, dV_dx3_R = secondOrderX3_4d(hcl_i, hcl_j, hcl_k, hcl_l, hcl_V, grid)
            dV_dx4_L, dV_dx4_R = secondOrderX4_4d(hcl_i, hcl_j, hcl_k, hcl_l, hcl_V, grid)

            hcl_deriv[0] = (dV_dx1_L + dV_dx1_R) / 2
            hcl_deriv[1] = (dV_dx2_L + dV_dx2_R) / 2
            hcl_deriv[2] = (dV_dx3_L + dV_dx3_R) / 2
            hcl_deriv[3] = (dV_dx4_L + dV_dx4_R) / 2

    hcl_deriv_ph = hcl.placeholder((DIM,), "deriv")
    hcl_V_ph = hcl.placeholder(V.shape, "V")
    hcl_index_ph = hcl.placeholder((DIM,), "index")
    s = hcl.create_schedule(
        [hcl_deriv_ph, hcl_V_ph, hcl_index_ph], hcl_spa_deriv_4d_schedule
    )
    hcl_spa_deriv_4d = hcl.build(s)

    hcl_spa_deriv = hcl.asarray(np.empty(DIM))
    hcl_V = hcl.asarray(V)

    def compute_spa_deriv(state):
        hcl_index = hcl.asarray(grid.get_index(state))
        hcl_spa_deriv_4d(hcl_spa_deriv, hcl_V, hcl_index)
        return hcl_spa_deriv.asnumpy()

    state = np.array([grid.grid_points[i][3] for i in range(DIM)])
    # index = grid.get_index(state)
    spa_deriv = compute_spa_deriv(state)
    ans = 2 * state
    print(spa_deriv, ans)

    assert np.allclose(spa_deriv, ans)

    # state = np.array([(grid.grid_points[i][3] + grid.grid_points[i][4]) / 2 for i in range(DIM)])
    # spa_deriv = compute_spa_deriv(state)
    # ans = 2 * state
    # print(spa_deriv, ans)

    # assert np.allclose(spa_deriv, ans)




def test_heterocl_spa_deriv_5d():
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    DIM = 5

    grid = Grid(
        np.array([-5 for _ in range(DIM)]),
        np.array([5 for _ in range(DIM)]),
        DIM,
        np.array([10 for _ in range(DIM)]),
    )
    pts = np.meshgrid(*grid.grid_points, indexing="ij", sparse=True)

    def f(x1, x2, x3, x4, x5):
        return x1**2 + x2**2 + x3**2 + x4**2 + x5**2

    V = f(*pts)
    # pick points at grid points

    def hcl_spa_deriv_5d_schedule(hcl_deriv, hcl_V, hcl_index):
        with hcl.Stage("Calculate Derivative"):
            # hcl statements only work with hcl tensors
            hcl_i = hcl.scalar(hcl_index[0], name="i")
            hcl_j = hcl.scalar(hcl_index[1], name="j")
            hcl_k = hcl.scalar(hcl_index[2], name="k")
            hcl_l = hcl.scalar(hcl_index[3], name="l")
            hcl_m = hcl.scalar(hcl_index[4], name="m")

            dV_dx1_L = hcl.scalar(0.0, "dV_dx1_L")
            dV_dx1_R = hcl.scalar(0.0, "dV_dx1_R")

            dV_dx2_L = hcl.scalar(0.0, "dV_dx2_L")
            dV_dx2_R = hcl.scalar(0.0, "dV_dx2_R")

            dV_dx3_L = hcl.scalar(0.0, "dV_dx3_L")
            dV_dx3_R = hcl.scalar(0.0, "dV_dx3_R")

            dV_dx4_L = hcl.scalar(0.0, "dV_dx4_L")
            dV_dx4_R = hcl.scalar(0.0, "dV_dx4_R")

            dV_dx5_L = hcl.scalar(0.0, "dV_dx5_L")
            dV_dx5_R = hcl.scalar(0.0, "dV_dx5_R")

            dV_dx1_L, dV_dx1_R = secondOrderX1_5d(
                hcl_i, hcl_j, hcl_k, hcl_l, hcl_m, hcl_V, grid
            )
            dV_dx2_L, dV_dx2_R = secondOrderX2_5d(
                hcl_i, hcl_j, hcl_k, hcl_l, hcl_m, hcl_V, grid
            )
            dV_dx3_L, dV_dx3_R = secondOrderX3_5d(
                hcl_i, hcl_j, hcl_k, hcl_l, hcl_m, hcl_V, grid
            )
            dV_dx4_L, dV_dx4_R = secondOrderX4_5d(
                hcl_i, hcl_j, hcl_k, hcl_l, hcl_m, hcl_V, grid
            )
            dV_dx5_L, dV_dx5_R = secondOrderX5_5d(
                hcl_i, hcl_j, hcl_k, hcl_l, hcl_m, hcl_V, grid
            )

            hcl_deriv[0] = (dV_dx1_L + dV_dx1_R) / 2
            hcl_deriv[1] = (dV_dx2_L + dV_dx2_R) / 2
            hcl_deriv[2] = (dV_dx3_L + dV_dx3_R) / 2
            hcl_deriv[3] = (dV_dx4_L + dV_dx4_R) / 2
            hcl_deriv[4] = (dV_dx5_L + dV_dx5_R) / 2

    hcl_deriv_ph = hcl.placeholder((DIM,), "deriv")
    hcl_V_ph = hcl.placeholder(V.shape, "V")
    hcl_index_ph = hcl.placeholder((DIM,), "index")
    s = hcl.create_schedule(
        [hcl_deriv_ph, hcl_V_ph, hcl_index_ph], hcl_spa_deriv_5d_schedule
    )
    hcl_spa_deriv_5d = hcl.build(s)

    hcl_spa_deriv = hcl.asarray(np.empty(DIM))
    hcl_V = hcl.asarray(V)

    def compute_spa_deriv(state):
        hcl_index = hcl.asarray(grid.get_index(state))
        hcl_spa_deriv_5d(hcl_spa_deriv, hcl_V, hcl_index)
        return hcl_spa_deriv.asnumpy()

    state = np.array([grid.grid_points[i][3] for i in range(DIM)])
    # index = grid.get_index(state)
    spa_deriv = compute_spa_deriv(state)
    ans = 2 * state
    print(spa_deriv, ans)

    assert np.allclose(spa_deriv, ans)

    # state = np.array([(grid.grid_points[i][3] + grid.grid_points[i][4]) / 2 for i in range(DIM)])
    # spa_deriv = compute_spa_deriv(state)
    # ans = 2 * state
    # print(spa_deriv, ans)

    # assert np.allclose(spa_deriv, ans)



def test_heterocl_spa_deriv_6d():
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    DIM = 6

    grid = Grid(
        np.array([-5 for _ in range(DIM)]),
        np.array([5 for _ in range(DIM)]),
        DIM,
        np.array([10 for _ in range(DIM)]),
    )
    pts = np.meshgrid(*grid.grid_points, indexing="ij", sparse=True)

    def f(x1, x2, x3, x4, x5, x6):
        return x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2

    V = f(*pts)

    def hcl_spa_deriv_6d_schedule(hcl_deriv, hcl_V, hcl_index):
        with hcl.Stage("Calculate Derivative"):
            # hcl statements only work with hcl tensors
            hcl_i = hcl.scalar(hcl_index[0], name="i")
            hcl_j = hcl.scalar(hcl_index[1], name="j")
            hcl_k = hcl.scalar(hcl_index[2], name="k")
            hcl_l = hcl.scalar(hcl_index[3], name="l")
            hcl_m = hcl.scalar(hcl_index[4], name="m")
            hcl_n = hcl.scalar(hcl_index[5], name="n")

            dV_dx1_L = hcl.scalar(0.0, "dV_dx1_L")
            dV_dx1_R = hcl.scalar(0.0, "dV_dx1_R")

            dV_dx2_L = hcl.scalar(0.0, "dV_dx2_L")
            dV_dx2_R = hcl.scalar(0.0, "dV_dx2_R")

            dV_dx3_L = hcl.scalar(0.0, "dV_dx3_L")
            dV_dx3_R = hcl.scalar(0.0, "dV_dx3_R")

            dV_dx4_L = hcl.scalar(0.0, "dV_dx4_L")
            dV_dx4_R = hcl.scalar(0.0, "dV_dx4_R")

            dV_dx5_L = hcl.scalar(0.0, "dV_dx5_L")
            dV_dx5_R = hcl.scalar(0.0, "dV_dx5_R")

            dV_dx6_L = hcl.scalar(0.0, "dV_dx6_L")
            dV_dx6_R = hcl.scalar(0.0, "dV_dx6_R")

            dV_dx1_L, dV_dx1_R = secondOrderX1_6d(
                hcl_i, hcl_j, hcl_k, hcl_l, hcl_m, hcl_n, hcl_V, grid
            )
            dV_dx2_L, dV_dx2_R = secondOrderX2_6d(
                hcl_i, hcl_j, hcl_k, hcl_l, hcl_m, hcl_n, hcl_V, grid
            )
            dV_dx3_L, dV_dx3_R = secondOrderX3_6d(
                hcl_i, hcl_j, hcl_k, hcl_l, hcl_m, hcl_n, hcl_V, grid
            )
            dV_dx4_L, dV_dx4_R = secondOrderX4_6d(
                hcl_i, hcl_j, hcl_k, hcl_l, hcl_m, hcl_n, hcl_V, grid
            )
            dV_dx5_L, dV_dx5_R = secondOrderX5_6d(
                hcl_i, hcl_j, hcl_k, hcl_l, hcl_m, hcl_n, hcl_V, grid
            )

            dV_dx6_L, dV_dx6_R = secondOrderX6_6d(
                hcl_i, hcl_j, hcl_k, hcl_l, hcl_m, hcl_n, hcl_V, grid
            )

            hcl_deriv[0] = (dV_dx1_L + dV_dx1_R) / 2
            hcl_deriv[1] = (dV_dx2_L + dV_dx2_R) / 2
            hcl_deriv[2] = (dV_dx3_L + dV_dx3_R) / 2
            hcl_deriv[3] = (dV_dx4_L + dV_dx4_R) / 2
            hcl_deriv[4] = (dV_dx5_L + dV_dx5_R) / 2
            hcl_deriv[5] = (dV_dx6_L + dV_dx6_R) / 2

    hcl_deriv_ph = hcl.placeholder((DIM,), "deriv")
    hcl_V_ph = hcl.placeholder(V.shape, "V")
    hcl_index_ph = hcl.placeholder((DIM,), "index")
    s = hcl.create_schedule(
        [hcl_deriv_ph, hcl_V_ph, hcl_index_ph], hcl_spa_deriv_6d_schedule
    )
    hcl_spa_deriv_6d = hcl.build(s)

    hcl_spa_deriv = hcl.asarray(np.empty(DIM))
    hcl_V = hcl.asarray(V)

    def compute_spa_deriv(state):
        hcl_index = hcl.asarray(grid.get_index(state))
        hcl_spa_deriv_6d(hcl_spa_deriv, hcl_V, hcl_index)
        return hcl_spa_deriv.asnumpy()

    state = np.array([grid.grid_points[i][3] for i in range(DIM)])
    # index = grid.get_index(state)
    spa_deriv = compute_spa_deriv(state)
    ans = 2 * state
    print(spa_deriv, ans)

    assert np.allclose(spa_deriv, ans)

    # state = np.array([(grid.grid_points[i][3] + grid.grid_points[i][4]) / 2 for i in range(DIM)])
    # spa_deriv = compute_spa_deriv(state)
    # ans = 2 * state
    # print(spa_deriv, ans)

    # assert np.allclose(spa_deriv, ans)