import heterocl as hcl
from odp.computeGraphs.CustomGraphFunctions import *

##############################  6D DERIVATIVE FUNCTIONS #############################
def spa_derivX6_6d(i, j, k, l, m, n, V, g):  # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(n == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n + 1] - V[i, j, k, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[5]
        right_deriv[0] = (V[i, j, k, l, m, n + 1] - V[i, j, k, l, m, n]) / g.dx[5]
    with hcl.elif_(n == V.shape[5] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i, j, k, l, m, n - 1]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l, m, n - 1]) / g.dx[5]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[5]
    with hcl.elif_(n != 0 and n != V.shape[5] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l, m, n - 1]) / g.dx[5]
        right_deriv[0] = (V[i, j, k, l, m, n + 1] - V[i, j, k, l, m, n]) / g.dx[5]
    return left_deriv[0], right_deriv[0]


def spa_derivX5_6d(i, j, k, l, m, n, V, g):  # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(m == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m + 1, n] - V[i, j, k, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[4]
        right_deriv[0] = (V[i, j, k, l, m + 1, n] - V[i, j, k, l, m, n]) / g.dx[4]
    with hcl.elif_(m == V.shape[4] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i, j, k, l, m - 1, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l, m - 1, n]) / g.dx[4]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[4]
    with hcl.elif_(m != 0 and m != V.shape[4] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l, m - 1, n]) / g.dx[4]
        right_deriv[0] = (V[i, j, k, l, m + 1, n] - V[i, j, k, l, m, n]) / g.dx[4]
    return left_deriv[0], right_deriv[0]


def spa_derivX4_6d(i, j, k, l, m, n, V, g):  # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(l == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l + 1, m, n] - V[i, j, k, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[3]
        right_deriv[0] = (V[i, j, k, l + 1, m, n] - V[i, j, k, l, m, n]) / g.dx[3]
    with hcl.elif_(l == V.shape[3] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i, j, k, l - 1, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l - 1, m, n]) / g.dx[3]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[3]
    with hcl.elif_(l != 0 and l != V.shape[3] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l - 1, m, n]) / g.dx[3]
        right_deriv[0] = (V[i, j, k, l + 1, m, n] - V[i, j, k, l, m, n]) / g.dx[3]
    return left_deriv[0], right_deriv[0]


def spa_derivX3_6d(i, j, k, l, m, n, V, g):  # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(k == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k + 1, l, m, n] - V[i, j, k, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[2]
        right_deriv[0] = (V[i, j, k + 1, l, m, n] - V[i, j, k, l, m, n]) / g.dx[2]
    with hcl.elif_(k == V.shape[2] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i, j, k - 1, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k - 1, l, m, n]) / g.dx[2]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[2]
    with hcl.elif_(k != 0 and k != V.shape[2] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k - 1, l, m, n]) / g.dx[2]
        right_deriv[0] = (V[i, j, k + 1, l, m, n] - V[i, j, k, l, m, n]) / g.dx[2]
    return left_deriv[0], right_deriv[0]


def spa_derivX2_6d(i, j, k, l, m, n, V, g):  # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(j == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j + 1, k, l, m, n] - V[i, j, k, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[1]
        right_deriv[0] = (V[i, j + 1, k, l, m, n] - V[i, j, k, l, m, n]) / g.dx[1]
    with hcl.elif_(j == V.shape[1] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i, j - 1, k, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j - 1, k, l, m, n]) / g.dx[1]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[1]
    with hcl.elif_(j != 0 and j != V.shape[1] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j - 1, k, l, m, n]) / g.dx[1]
        right_deriv[0] = (V[i, j + 1, k, l, m, n] - V[i, j, k, l, m, n]) / g.dx[1]
    return left_deriv[0], right_deriv[0]


def spa_derivX1_6d(i, j, k, l, m, n, V, g):  # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(i == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i + 1, j, k, l, m, n] - V[i, j, k, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[0]
        right_deriv[0] = (V[i + 1, j, k, l, m, n] - V[i, j, k, l, m, n]) / g.dx[0]
    with hcl.elif_(i == V.shape[0] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i - 1, j, k, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i - 1, j, k, l, m, n]) / g.dx[0]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[0]
    with hcl.elif_(i != 0 and i != V.shape[0] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i - 1, j, k, l, m, n]) / g.dx[0]
        right_deriv[0] = (V[i + 1, j, k, l, m, n] - V[i, j, k, l, m, n]) / g.dx[0]
    return left_deriv[0], right_deriv[0]
