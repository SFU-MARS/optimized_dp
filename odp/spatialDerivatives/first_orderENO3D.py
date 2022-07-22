import heterocl as hcl
from odp.computeGraphs.CustomGraphFunctions import *

################## 3D SPATIAL DERIVATIVE FUNCTION #################
def spa_derivX(i, j, k, V, g):
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    if 0 not in g.pDim:
        with hcl.if_(i == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, j, k] + my_abs(V[i + 1, j, k] - V[i, j, k]) * my_sign(\
                V[i, j, k])
            left_deriv[0] = (V[i, j, k] - left_boundary[0]) / g.dx[0]
            right_deriv[0] = (V[i + 1, j, k] - V[i, j, k]) / g.dx[0]
        with hcl.elif_(i == V.shape[0] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, j, k] + my_abs(V[i, j, k] - V[i - 1, j, k]) * my_sign(
                V[i, j, k])
            left_deriv[0] = (V[i, j, k] - V[i - 1, j, k]) / g.dx[0]
            right_deriv[0] = (right_boundary[0] - V[i, j, k]) / g.dx[0]
        with hcl.elif_(i != 0 and i != V.shape[0] - 1):
            left_deriv[0] = (V[i, j, k] - V[i - 1, j, k]) / g.dx[0]
            right_deriv[0] = (V[i + 1, j, k] - V[i, j, k]) / g.dx[0]
        return left_deriv[0], right_deriv[0]
    else:
        with hcl.if_(i == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[V.shape[0] - 1, j, k]
            left_deriv[0] = (V[i, j, k] - left_boundary[0]) / g.dx[0]
            right_deriv[0] = (V[i + 1, j, k] - V[i, j, k]) / g.dx[0]
        with hcl.elif_(i == V.shape[0] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[0, j, k]
            left_deriv[0] = (V[i, j, k] - V[i - 1, j, k]) / g.dx[0]
            right_deriv[0] = (right_boundary[0] - V[i, j, k]) / g.dx[0]
        with hcl.elif_(i != 0 and i != V.shape[0] - 1):
            left_deriv[0] = (V[i, j, k] - V[i - 1, j, k]) / g.dx[0]
            right_deriv[0] = (V[i + 1, j, k] - V[i, j, k]) / g.dx[0]
        return left_deriv[0], right_deriv[0]


def spa_derivY(i, j, k, V, g):
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    if 1 not in g.pDim:
        with hcl.if_(j == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, j, k] + my_abs(V[i, j + 1, k] - V[i, j, k]) * my_sign(
                V[i, j, k])
            left_deriv[0] = (V[i, j, k] - left_boundary[0]) / g.dx[1]
            right_deriv[0] = (V[i, j + 1, k] - V[i, j, k]) / g.dx[1]
        with hcl.elif_(j == V.shape[1] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, j, k] + my_abs(V[i, j, k] - V[i, j - 1, k]) * my_sign(
                V[i, j, k])
            left_deriv[0] = (V[i, j, k] - V[i, j - 1, k]) / g.dx[1]
            right_deriv[0] = (right_boundary[0] - V[i, j, k]) / g.dx[1]
        with hcl.elif_(j != 0 and j != V.shape[1] - 1):
            left_deriv[0] = (V[i, j, k] - V[i, j - 1, k]) / g.dx[1]
            right_deriv[0] = (V[i, j + 1, k] - V[i, j, k]) / g.dx[1]
        return left_deriv[0], right_deriv[0]
    else:
        with hcl.if_(j == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, V.shape[1] - 1, k]
            left_deriv[0] = (V[i, j, k] - left_boundary[0]) / g.dx[1]
            right_deriv[0] = (V[i, j + 1, k] - V[i, j, k]) / g.dx[1]
        with hcl.elif_(j == V.shape[1] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, 0, k]
            left_deriv[0] = (V[i, j, k] - V[i, j - 1, k]) / g.dx[1]
            right_deriv[0] = (right_boundary[0] - V[i, j, k]) / g.dx[1]
        with hcl.elif_(j != 0 and j != V.shape[1] - 1):
            left_deriv[0] = (V[i, j, k] - V[i, j - 1, k]) / g.dx[1]
            right_deriv[0] = (V[i, j + 1, k] - V[i, j, k]) / g.dx[1]
        return left_deriv[0], right_deriv[0]


def spa_derivT(i, j, k, V, g):
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    if 2 not in g.pDim:
        with hcl.if_(k == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, j, k] + my_abs(V[i, j, k + 1] - V[i, j, k]) * my_sign(
            V[i, j, k])
            left_deriv[0] = (V[i, j, k] - left_boundary[0]) / g.dx[2]
            right_deriv[0] = (V[i, j, k + 1] - V[i, j, k]) / g.dx[2]
        with hcl.elif_(k == V.shape[2] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, j, k] + my_abs(V[i, j, k] - V[i, j, k - 1]) * my_sign(
            V[i, j, k])
            left_deriv[0] = (V[i, j, k] - V[i, j, k - 1]) / g.dx[2]
            right_deriv[0] = (right_boundary[0] - V[i, j, k]) / g.dx[2]
        with hcl.elif_(k != 0 and k != V.shape[2] - 1):
            left_deriv[0] = (V[i, j, k] - V[i, j, k - 1]) / g.dx[2]
            right_deriv[0] = (V[i, j, k + 1] - V[i, j, k]) / g.dx[2]
        return left_deriv[0], right_deriv[0]
    else:
        with hcl.if_(k == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, j, V.shape[2] - 1]
            left_deriv[0] = (V[i, j, k] - left_boundary[0]) / g.dx[2]
            right_deriv[0] = (V[i, j, k + 1] - V[i, j, k]) / g.dx[2]
        with hcl.elif_(k == V.shape[2] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, j, 0]
            left_deriv[0] = (V[i, j, k] - V[i, j, k - 1]) / g.dx[2]
            right_deriv[0] = (right_boundary[0] - V[i, j, k]) / g.dx[2]
        with hcl.elif_(k != 0 and k != V.shape[2] - 1):
            left_deriv[0] = (V[i, j, k] - V[i, j, k - 1]) / g.dx[2]
            right_deriv[0] = (V[i, j, k + 1] - V[i, j, k]) / g.dx[2]
        return left_deriv[0], right_deriv[0]