import heterocl as hcl
from odp.computeGraphs.CustomGraphFunctions import *


################## 4D SPATIAL DERIVATIVE FUNCTION #################

# Calculate derivative on the first derivative  #
def spa_derivX4_4d(i, j, k, l, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    if 3 not in g.pDim:
        with hcl.if_(l == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, j, k, l] + my_abs(V[i, j, k, l + 1] - V[i, j, k, l]) * my_sign(
                V[i, j, k, l])
            left_deriv[0] = (V[i, j, k, l] - left_boundary[0]) / g.dx[3]
            right_deriv[0] = (V[i, j, k, l + 1] - V[i, j, k, l]) / g.dx[3]
        with hcl.elif_(l == V.shape[3] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, j, k, l] + my_abs(V[i, j, k, l] - V[i, j, k, l - 1]) * my_sign(
                V[i, j, k, l])
            left_deriv[0] = (V[i, j, k, l] - V[i, j, k, l - 1]) / g.dx[3]
            right_deriv[0] = (right_boundary[0] - V[i, j, k, l]) / g.dx[3]
        with hcl.elif_(l != 0 and l != V.shape[3] - 1):
            left_deriv[0] = (V[i, j, k, l] - V[i, j, k, l - 1]) / g.dx[3]
            right_deriv[0] = (V[i, j, k, l + 1] - V[i, j, k, l]) / g.dx[3]
        return left_deriv[0], right_deriv[0]
    else:
        with hcl.if_(l == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, j, k, V.shape[3] - 1]
            left_deriv[0] = (V[i, j, k, l] - left_boundary[0]) / g.dx[3]
            right_deriv[0] = (V[i, j, k, l + 1] - V[i, j, k, l]) / g.dx[3]
        with hcl.elif_(l == V.shape[3] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, j, k, 0]
            left_deriv[0] = (V[i, j, k , l] - V[i, j, k, l - 1]) / g.dx[3]
            right_deriv[0] = (right_boundary[0] - V[i, j, k, l]) / g.dx[3]
        with hcl.elif_(l != 0 and l != V.shape[3] - 1):
            left_deriv[0] = (V[i, j, k, l] - V[i, j, k, l - 1]) / g.dx[3]
            right_deriv[0] = (V[i, j, k, l + 1] - V[i, j, k, l]) / g.dx[3]
        return left_deriv[0], right_deriv[0]


def spa_derivX3_4d(i, j, k, l, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    if 2 not in g.pDim:
        with hcl.if_(k == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, j, k, l] + my_abs(V[i, j, k + 1, l ] - V[i, j, k, l]) * my_sign(
                V[i, j, k, l])
            left_deriv[0] = (V[i, j, k, l] - left_boundary[0]) / g.dx[2]
            right_deriv[0] = (V[i, j, k + 1, l] - V[i, j, k, l]) / g.dx[2]
        with hcl.elif_(k == V.shape[2] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, j, k, l] + my_abs(V[i, j, k, l] - V[i, j, k - 1, l]) * my_sign(
                V[i, j, k, l])
            left_deriv[0] = (V[i, j, k, l] - V[i, j, k - 1, l]) / g.dx[2]
            right_deriv[0] = (right_boundary[0] - V[i, j, k, l]) / g.dx[2]
        with hcl.elif_(k != 0 and k != V.shape[2] - 1):
            left_deriv[0] = (V[i, j, k, l] - V[i, j, k - 1, l]) / g.dx[2]
            right_deriv[0] = (V[i, j, k + 1, l] - V[i, j, k, l]) / g.dx[2]
        return left_deriv[0], right_deriv[0]
    else:
        with hcl.if_(k == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, j, V.shape[2] - 1, l]
            left_deriv[0] = (V[i, j, k, l] - left_boundary[0]) / g.dx[2]
            right_deriv[0] = (V[i, j, k + 1, l] - V[i, j, k, l]) / g.dx[2]
        with hcl.elif_(k == V.shape[2] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, j, 0, l]
            left_deriv[0] = (V[i, j, k , l] - V[i, j, k - 1, l]) / g.dx[2]
            right_deriv[0] = (right_boundary[0] - V[i, j, k, l]) / g.dx[2]
        with hcl.elif_(k != 0 and k != V.shape[2] - 1):
            left_deriv[0] = (V[i, j, k, l] - V[i, j, k - 1, l]) / g.dx[2]
            right_deriv[0] = (V[i, j, k + 1, l] - V[i, j, k, l]) / g.dx[2]
        return left_deriv[0], right_deriv[0]

def spa_derivX2_4d(i, j, k, l, V, g): #
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    if 1 not in g.pDim:
        with hcl.if_(j == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, j, k, l] + my_abs(V[i, j + 1, k, l] - V[i, j, k, l]) * my_sign(
                V[i, j, k, l])
            left_deriv[0] = (V[i, j, k, l] - left_boundary[0]) / g.dx[1]
            right_deriv[0] = (V[i, j + 1, k, l] - V[i, j, k, l]) / g.dx[1]
        with hcl.elif_(j == V.shape[1] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, j, k, l] + my_abs(V[i, j, k, l] - V[i, j - 1, k, l]) * my_sign(
                V[i, j, k, l])
            left_deriv[0] = (V[i, j, k, l] - V[i, j - 1, k, l]) / g.dx[1]
            right_deriv[0] = (right_boundary[0] - V[i, j, k, l]) / g.dx[1]
        with hcl.elif_(j != 0 and j != V.shape[1] - 1):
            left_deriv[0] = (V[i, j, k, l] - V[i, j - 1, k, l]) / g.dx[1]
            right_deriv[0] = (V[i, j + 1, k, l] - V[i, j, k, l]) / g.dx[1]
        return left_deriv[0], right_deriv[0]
    else:
        with hcl.if_(j == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, V.shape[1] - 1 , k, l]
            left_deriv[0] = (V[i, j, k, l] - left_boundary[0]) / g.dx[1]
            right_deriv[0] = (V[i, j + 1, k, l] - V[i, j, k, l]) / g.dx[1]
        with hcl.elif_(j == V.shape[1] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, 0, k, l]
            left_deriv[0] = (V[i, j, k, l] - V[i, j - 1, k, l]) / g.dx[1]
            right_deriv[0] = (right_boundary[0] - V[i, j, k, l]) / g.dx[1]
        with hcl.elif_(j != 0 and j != V.shape[1] - 1):
            left_deriv[0] = (V[i, j, k, l] - V[i, j - 1, k, l]) / g.dx[1]
            right_deriv[0] = (V[i, j + 1, k, l] - V[i, j, k, l]) / g.dx[1]
        return left_deriv[0], right_deriv[0]

def spa_derivX1_4d(i, j, k, l, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    if 0 not in g.pDim:
        with hcl.if_(i == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i, j, k, l] + my_abs(V[i + 1, j, k, l] - V[i, j, k, l]) * my_sign(
                V[i, j, k, l])
            left_deriv[0] = (V[i, j, k, l] - left_boundary[0]) / g.dx[0]
            right_deriv[0] = (V[i + 1, j, k, l] - V[i, j, k, l]) / g.dx[0]
        with hcl.elif_(i == V.shape[0] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i, j, k, l] + my_abs(V[i, j, k, l] - V[i - 1, j, k, l]) * my_sign(
                V[i, j, k, l])
            left_deriv[0] = (V[i, j, k, l] - V[i - 1, j, k, l]) / g.dx[0]
            right_deriv[0] = (right_boundary[0] - V[i, j, k, l]) / g.dx[0]
        with hcl.elif_(i != 0 and i != V.shape[0] - 1):
            left_deriv[0] = (V[i, j, k, l] - V[i -1, j, k, l]) / g.dx[0]
            right_deriv[0] = (V[i + 1, j, k, l] - V[i, j, k, l]) / g.dx[0]
        return left_deriv[0], right_deriv[0]
    else:
        with hcl.if_(i == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[V.shape[0] - 1, j, k, l]
            left_deriv[0] = (V[i, j, k, l] - left_boundary[0]) / g.dx[0]
            right_deriv[0] = (V[i + 1, j, k, l] - V[i, j, k, l]) / g.dx[0]
        with hcl.elif_(i == V.shape[0] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[0, j, k, l]
            left_deriv[0] = (V[i, j, k, l] - V[i - 1, j, k, l]) / g.dx[0]
            right_deriv[0] = (right_boundary[0] - V[i, j, k, l]) / g.dx[0]
        with hcl.elif_(i != 0 and i != V.shape[0] - 1):
            left_deriv[0] = (V[i, j, k, l] - V[i -1, j, k, l]) / g.dx[0]
            right_deriv[0] = (V[i + 1, j, k, l] - V[i, j, k, l]) / g.dx[0]
        return left_deriv[0], right_deriv[0]