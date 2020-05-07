import heterocl as hcl
import numpy as np
import time
from user_definer import *
# Custom function

def my_min(a,b):
    result = hcl.scalar(0, "result")
    with hcl.if_(a < b):
        result[0] = a
    with hcl.else_():
        result[0] = b
    return result[0]

# Input is value, output is value
def my_max(a,b):
    result = hcl.scalar(0, "result")
    with hcl.if_(a > b):
        result.v = a
    with hcl.else_():
        result.v = b
    return result.v

# Input is value, output is value
def max_4n(num1, num2, num3, num4):
    largest = hcl.scalar(0, "largest")
    largest[0] = my_max(num1, num2)
    largest[0] = my_max(largest[0], num3)
    largest[0] = my_max(largest[0], num4)

    return largest[0]

# Input is a value, output is value
def my_abs(my_x):
    abs_value = hcl.scalar(0, "abs_value", dtype=hcl.Float())
    with hcl.if_(my_x > 0):
        abs_value.v = my_x
    with hcl.else_():
        abs_value.v = -my_x
    return abs_value.v

def new_abs(my_x):
    my_value = hcl.scalar(0, "my_value", dtype=hcl.Float())
    with hcl.if_(my_x > 0):
        my_value.v = my_x
    with hcl.else_():
        my_value.v = -my_x
    return my_value.v

def my_sign(x):
    sign = hcl.scalar(0, "sign", dtype=hcl.Float())
    with hcl.if_(x == 0):
        sign[0] = 0
    with hcl.if_(x > 0):
        sign[0] = 1
    with hcl.if_(x < 0):
        sign[0] = -1
    return sign[0]

########################## 3D SPATIAL DERIVATIVE FUNCTION #################################

def spa_derivX(i, j, k, V, g):
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(i == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[k, j, i] + my_abs(V[k, j, i + 1] - V[k, j, i]) * my_sign(
            V[k, j, i])
        left_deriv[0] = (V[k, j, i] - left_boundary[0]) / g.dx[0]
        right_deriv[0] = (V[k, j, i + 1] - V[k, j, i]) / g.dx[0]
    with hcl.elif_(i == V.shape[2] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[k, j, i] + my_abs(V[k, j, i] - V[k, j, i - 1]) * my_sign(
            V[k, j, i])
        left_deriv[0] = (V[k, j, i] - V[k, j, i - 1]) / g.dx[0]
        right_deriv[0] = (right_boundary[0] - V[k, j, i]) / g.dx[0]
    with hcl.elif_(i != 0 and i != V.shape[2] - 1):
        left_deriv[0] = (V[k, j, i] - V[k, j, i-1]) / g.dx[0]
        right_deriv[0] = (V[k, j, i+1] - V[k, j, i]) / g.dx[0]
    return left_deriv[0], right_deriv[0]


def spa_derivY(i, j, k, V, g):
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(j == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[k, j, i] + my_abs(V[k, j + 1, i] - V[k, j, i]) * my_sign(
            V[k, j, i])
        left_deriv[0] = (V[k, j, i] - left_boundary[0]) / g.dx[1]
        right_deriv[0] = (V[k, j + 1, i] - V[k, j, i]) / g.dx[1]
    with hcl.elif_(j == V.shape[1] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[k, j, i] + my_abs(V[k, j, i] - V[k, j - 1, i]) * my_sign(
            V[k, j, i])
        left_deriv[0] = (V[k, j, i] - V[k, j - 1, i]) / g.dx[1]
        right_deriv[0] = (right_boundary[0] - V[k, j, i]) / g.dx[1]
    with hcl.elif_(j != 0 and j != V.shape[1] - 1):
        left_deriv[0] = (V[k, j, i] - V[k, j - 1, i]) / g.dx[1]
        right_deriv[0] = (V[k, j + 1, i] - V[k, j, i]) / g.dx[1]
    return left_deriv[0], right_deriv[0]

def spa_derivT(i, j, k, V, g):
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(k == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        # left_boundary[0] = V_init[i,j,50]
        left_boundary[0] = V[V.shape[0] - 1, j, i]
        left_deriv[0] = (V[k, j, i] - left_boundary[0]) / g.dx[2]
        right_deriv[0] = (V[k + 1, j, i] - V[k, j, i]) / g.dx[2]
    with hcl.elif_(k == V.shape[0] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[0, j, i]
        left_deriv[0] = (V[k, j, i] - V[k-1, j, i]) / g.dx[2]
        right_deriv[0] = (right_boundary[0] - V[k, j, i]) / g.dx[2]
    with hcl.elif_(k != 0 and k != V.shape[0] - 1):
        left_deriv[0] = (V[k, j, i] - V[k-1, j, i]) / g.dx[2]
        right_deriv[0] = (V[k+1, j, i] - V[k, j, i]) / g.dx[2]
    return left_deriv[0], right_deriv[0]

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


##############################  6D DERIVATIVE FUNCTIONS #############################
def spa_derivX1_6d(i, j, k, l, m, n, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(n == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n + 1] - V[i, j, k, l, m, n]) * my_sign(V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[5]
        right_deriv[0] = (V[i, j, k, l, m, n + 1] - V[i, j, k, l, m, n]) / g.dx[5]
    with hcl.elif_(n == V.shape[5] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i, j, k, l, m, n-1]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l, m, n - 1]) / g.dx[5]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[5]
    with hcl.elif_(n != 0 and n != V.shape[5] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l, m, n -1]) / g.dx[5]
        right_deriv[0] = (V[i, j, k, l, m, n + 1] - V[i, j, k, l, m, n]) / g.dx[5]
    return left_deriv[0], right_deriv[0]

def spa_derivX2_6d(i, j, k, l, m, n, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(m == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m + 1, n] - V[i, j, k, l, m, n]) * my_sign(V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[4]
        right_deriv[0] = (V[i, j, k, l, m + 1, n] - V[i, j, k, l, m, n]) / g.dx[4]
    with hcl.elif_(m == V.shape[4] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i, j, k, l, m - 1, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l, m -1 , n]) / g.dx[4]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[4]
    with hcl.elif_(m != 0 and m != V.shape[4] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l, m - 1, n]) / g.dx[4]
        right_deriv[0] = (V[i, j, k, l, m + 1, n] - V[i, j, k, l, m, n]) / g.dx[4]
    return left_deriv[0], right_deriv[0]

def spa_derivX3_6d(i, j, k, l, m, n, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(l == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l + 1, m, n] - V[i, j, k, l, m, n]) * my_sign(V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[3]
        right_deriv[0] = (V[i, j, k, l + 1,m, n] - V[i, j, k, l, m, n]) / g.dx[3]
    with hcl.elif_(l == V.shape[3] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i, j, k, l - 1, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l-1, m , n]) / g.dx[3]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[3]
    with hcl.elif_(l != 0 and l != V.shape[3] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k, l-1, m, n]) / g.dx[3]
        right_deriv[0] = (V[i, j, k, l + 1, m, n] - V[i, j, k, l, m, n]) / g.dx[3]
    return left_deriv[0], right_deriv[0]

def spa_derivX4_6d(i, j, k, l, m, n, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(k == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k + 1, l, m, n] - V[i, j, k, l, m, n]) * my_sign(V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[2]
        right_deriv[0] = (V[i, j, k + 1, l, m, n] - V[i, j, k, l, m, n]) / g.dx[2]
    with hcl.elif_(k == V.shape[2] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i, j, k-1, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k-1, l, m, n]) / g.dx[2]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[2]
    with hcl.elif_(k != 0 and k != V.shape[2] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j, k - 1, l, m, n]) / g.dx[2]
        right_deriv[0] = (V[i, j, k + 1, l, m, n] - V[i, j, k, l, m, n]) / g.dx[2]
    return left_deriv[0], right_deriv[0]

def spa_derivX5_6d(i, j, k, l, m, n, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(j == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j + 1, k, l, m, n] - V[i, j, k, l, m, n]) * my_sign(V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[1]
        right_deriv[0] = (V[i, j + 1, k, l, m, n] - V[i, j, k, l, m, n]) / g.dx[1]
    with hcl.elif_(j == V.shape[1] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i, j -1 , k, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j - 1, k, l, m, n]) / g.dx[1]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[1]
    with hcl.elif_(j != 0 and j != V.shape[1] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i, j - 1, k, l, m, n]) / g.dx[1]
        right_deriv[0] = (V[i, j + 1, k, l, m, n] - V[i, j, k, l, m, n]) / g.dx[1]
    return left_deriv[0], right_deriv[0]

def spa_derivX6_6d(i, j, k, l, m, n, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    with hcl.if_(i == 0):
        left_boundary = hcl.scalar(0, "left_boundary")
        left_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i + 1, j, k, l, m, n] - V[i, j, k, l, m, n]) * my_sign(V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - left_boundary[0]) / g.dx[0]
        right_deriv[0] = (V[i + 1, j, k, l, m, n] - V[i, j, k, l, m, n]) / g.dx[0]
    with hcl.elif_(i == V.shape[0] - 1):
        right_boundary = hcl.scalar(0, "right_boundary")
        right_boundary[0] = V[i, j, k, l, m, n] + my_abs(V[i, j, k, l, m, n] - V[i-1, j, k, l, m, n]) * my_sign(
            V[i, j, k, l, m, n])
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i-1, j, k, l, m, n]) / g.dx[0]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l, m, n]) / g.dx[0]
    with hcl.elif_(i != 0 and i != V.shape[0] - 1):
        left_deriv[0] = (V[i, j, k, l, m, n] - V[i - 1, j, k, l, m, n]) / g.dx[0]
        right_deriv[0] = (V[i + 1, j, k, l, m, n] - V[i, j, k, l, m, n]) /g.dx[0]
    return left_deriv[0], right_deriv[0]

########################## 6D graph definition ########################

# Note that t has 2 elements t1, t2
def graph_6D(V_new, V_init, deriv_diff1, deriv_diff2, deriv_diff3, deriv_diff4, deriv_diff5, deriv_diff6,
             x1, x2, x3, x4, x5, x6 ,t , l0):
    # Maximum derivative for each dim
    max_deriv1 = hcl.scalar(-1e9, "max_deriv1")
    max_deriv2 = hcl.scalar(-1e9, "max_deriv2")
    max_deriv3 = hcl.scalar(-1e9, "max_deriv3")
    max_deriv4 = hcl.scalar(-1e9, "max_deriv4")
    max_deriv5 = hcl.scalar(-1e9, "max_deriv5")
    max_deriv6 = hcl.scalar(-1e9, "max_deriv6")
    
    # Min derivative for each dim
    min_deriv1 = hcl.scalar(1e9, "min_deriv1")
    min_deriv2 = hcl.scalar(1e9, "min_deriv2")
    min_deriv3 = hcl.scalar(1e9, "min_deriv3")
    min_deriv4 = hcl.scalar(1e9, "min_deriv4")
    min_deriv5 = hcl.scalar(1e9, "min_deriv5")
    min_deriv6 = hcl.scalar(1e9, "min_deriv6")
    
    # These variables are used to dissipation calculation
    max_alpha1 = hcl.scalar(-1e9, "max_alpha1")
    max_alpha2 = hcl.scalar(-1e9, "max_alpha2")
    max_alpha3 = hcl.scalar(-1e9, "max_alpha3")
    max_alpha4 = hcl.scalar(-1e9, "max_alpha4")
    max_alpha5 = hcl.scalar(-1e9, "max_alpha5")
    max_alpha6 = hcl.scalar(-1e9, "max_alpha6")
    
    def step_bound(): # Function to calculate time step
        stepBoundInv = hcl.scalar(0, "stepBoundInv")
        stepBound    = hcl.scalar(0, "stepBound")
        stepBoundInv[0] = max_alpha1[0]/g.dx[0] + max_alpha2[0]/g.dx[1] + max_alpha3[0]/g.dx[2] + max_alpha4[0]/g.dx[3] \
                            + max_alpha5[0]/g.dx[4] + max_alpha6[0]/g.dx[5]

        stepBound[0] = 0.8/stepBoundInv[0]
        with hcl.if_(stepBound > t[1] - t[0]):
            stepBound[0] = t[1] - t[0]

        # Update the lower time ranges
        t[0] = t[0] + stepBound[0]
        #t[0] = min_deriv2[0]
        return stepBound[0]

    def maxVWithV0(i, j, k, l, m, n): # Take the max
        with hcl.if_(V_new[i, j, k, l, m, n] < l0[i, j, k, l, m, n]):
            V_new[i, j, k, l, m, n] = l0[i, j, k, l, m, n]

    # Max(V, g )
    def maxVWithCStraint(i, j, k, l, m, n):
        with hcl.if_(V_new[i, j, k, l, m, n] < 5.0):
            V_new[i, j, k, l, m, n] = 1.0

    # Min with V_before
    def minVWithVInit(i, j, k, l, m, n):
        with hcl.if_(V_new[i, j, k, l, m, n] > V_init[i, j, k, l, m, n]):
            V_new[i, j, k, l, m, n] = V_init[i, j, k, l, m, n]
                
    # Calculate Hamiltonian for every grid point in V_init
    with hcl.Stage("Hamiltonian"):
        with hcl.for_(0, V_init.shape[0], name="i") as i:
            with hcl.for_(0, V_init.shape[1], name="j") as j:
                with hcl.for_(0, V_init.shape[2], name="k") as k:
                    with hcl.for_(0, V_init.shape[3], name="l") as l:
                        with hcl.for_(0, V_init.shape[4], name="m") as m:
                            with hcl.for_(0, V_init.shape[5], name="n") as n:
                                #Variables to calculate dV_dx
                                dV_dx1_L = hcl.scalar(0, "dV_dx1_L")
                                dV_dx1_R = hcl.scalar(0, "dV_dx1_R")
                                dV_dx1 = hcl.scalar(0, "dV_dx1")
                                dV_dx2_L = hcl.scalar(0, "dV_dx2_L")
                                dV_dx2_R = hcl.scalar(0, "dV_dx2_R")
                                dV_dx2 = hcl.scalar(0, "dV_dx2")
                                dV_dx3_L = hcl.scalar(0, "dV_dx3_L")
                                dV_dx3_R = hcl.scalar(0, "dV_dx3_R")
                                dV_dx3 = hcl.scalar(0, "dV_dx3")
                                dV_dx4_L = hcl.scalar(0, "dV_dx4_L")
                                dV_dx4_R = hcl.scalar(0, "dV_dx4_R")
                                dV_dx4 = hcl.scalar(0, "dV_dx4")
                                dV_dx5_L = hcl.scalar(0, "dV_dx5_L")
                                dV_dx5_R = hcl.scalar(0, "dV_dx5_R")
                                dV_dx5 = hcl.scalar(0, "dV_dx5")
                                dV_dx6_L = hcl.scalar(0, "dV_dx6_L")
                                dV_dx6_R = hcl.scalar(0, "dV_dx6_R")
                                dV_dx6 = hcl.scalar(0, "dV_dx6")

                                # No tensor slice operation
                                #dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
                                dV_dx1_L[0], dV_dx1_R[0] = spa_derivX6_6d(i, j, k, l, m, n, V_init, g)
                                dV_dx2_L[0], dV_dx2_R[0] = spa_derivX5_6d(i, j, k, l, m, n, V_init, g)
                                dV_dx3_L[0], dV_dx3_R[0] = spa_derivX4_6d(i, j, k, l, m, n, V_init, g)
                                dV_dx4_L[0], dV_dx4_R[0] = spa_derivX3_6d(i, j, k, l, m, n, V_init, g)
                                dV_dx5_L[0], dV_dx5_R[0] = spa_derivX2_6d(i, j, k, l, m, n, V_init, g)
                                dV_dx6_L[0], dV_dx6_R[0] = spa_derivX1_6d(i, j, k, l, m, n, V_init, g)

                                # Saves spatial derivative diff into tables
                                deriv_diff1[i, j, k, l, m, n] = dV_dx1_R[0] - dV_dx1_L[0]
                                deriv_diff2[i, j, k, l, m, n] = dV_dx2_R[0] - dV_dx2_L[0]
                                deriv_diff3[i, j, k, l, m, n] = dV_dx3_R[0] - dV_dx3_L[0]
                                deriv_diff4[i, j, k, l, m, n] = dV_dx4_R[0] - dV_dx4_L[0]
                                deriv_diff5[i, j, k, l, m, n] = dV_dx5_R[0] - dV_dx5_L[0]
                                deriv_diff6[i, j, k, l, m, n] = dV_dx6_R[0] - dV_dx6_L[0]

                                #Calculate average gradient
                                dV_dx1[0] = (dV_dx1_L + dV_dx1_R) / 2
                                dV_dx2[0] = (dV_dx2_L + dV_dx2_R) / 2
                                dV_dx3[0] = (dV_dx3_L + dV_dx3_R) / 2
                                dV_dx4[0] = (dV_dx4_L + dV_dx4_R) / 2
                                dV_dx5[0] = (dV_dx5_L + dV_dx5_R) / 2
                                dV_dx6[0] = (dV_dx6_L + dV_dx6_R) / 2


                                # Find optimal control
                                uOpt = my_object.opt_ctrl(t,(x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), (dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0], dV_dx5[0], dV_dx6[0]))
                                # Find optimal disturbance
                                dOpt = my_object.optDstb((dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0], dV_dx5[0], dV_dx6[0]))

                                # Find rates of changes based on dynamics equation
                                dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt, dx6_dt = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), uOpt, dOpt)

                                # Calculate Hamiltonian terms:
                                V_new[i, j, k, l, m, n] = -(dx1_dt * dV_dx1[0] + dx2_dt * dV_dx2[0] + dx3_dt * dV_dx3[0] + dx4_dt * dV_dx4[0] + dx5_dt * dV_dx5[0] + dx6_dt * dV_dx6[0])
                                
                                # Get derivMin
                                with hcl.if_(dV_dx1_L[0] < min_deriv1[0]):
                                    min_deriv1[0] = dV_dx1_L[0]
                                with hcl.if_(dV_dx1_R[0] < min_deriv1[0]):
                                    min_deriv1[0] = dV_dx1_R[0]

                                with hcl.if_(dV_dx2_L[0] < min_deriv2[0]):
                                    min_deriv2[0] = dV_dx2_L[0]
                                with hcl.if_(dV_dx2_R[0] < min_deriv2[0]):
                                    min_deriv2[0] = dV_dx2_R[0]

                                with hcl.if_(dV_dx3_L[0] < min_deriv3[0]):
                                    min_deriv3[0] = dV_dx3_L[0]
                                with hcl.if_(dV_dx3_R[0] < min_deriv3[0]):
                                    min_deriv3[0] = dV_dx3_R[0]

                                with hcl.if_(dV_dx4_L[0] < min_deriv4[0]):
                                    min_deriv4[0] = dV_dx4_L[0]
                                with hcl.if_(dV_dx4_R[0] < min_deriv4[0]):
                                    min_deriv4[0] = dV_dx4_R[0]

                                with hcl.if_(dV_dx5_L[0] < min_deriv5[0]):
                                    min_deriv5[0] = dV_dx5_L[0]
                                with hcl.if_(dV_dx5_R[0] < min_deriv5[0]):
                                    min_deriv5[0] = dV_dx5_R[0]

                                with hcl.if_(dV_dx6_L[0] < min_deriv6[0]):
                                    min_deriv6[0] = dV_dx6_L[0]
                                with hcl.if_(dV_dx6_R[0] < min_deriv6[0]):
                                    min_deriv6[0] = dV_dx6_R[0]

                                # Get derivMax
                                with hcl.if_(dV_dx1_L[0] > max_deriv1[0]):
                                    max_deriv1[0] = dV_dx1_L[0]
                                with hcl.if_(dV_dx1_R[0] > max_deriv1[0]):
                                    max_deriv1[0] = dV_dx1_R[0]

                                with hcl.if_(dV_dx2_L[0] > max_deriv2[0]):
                                    max_deriv2[0] = dV_dx2_L[0]
                                with hcl.if_(dV_dx2_R[0] > max_deriv2[0]):
                                    max_deriv2[0] = dV_dx2_R[0]

                                with hcl.if_(dV_dx3_L[0] > max_deriv3[0]):
                                    max_deriv3[0] = dV_dx3_L[0]
                                with hcl.if_(dV_dx3_R[0] > max_deriv3[0]):
                                    max_deriv3[0] = dV_dx3_R[0]

                                with hcl.if_(dV_dx4_L[0] > max_deriv4[0]):
                                    max_deriv4[0] = dV_dx4_L[0]
                                with hcl.if_(dV_dx4_R[0] > max_deriv4[0]):
                                    max_deriv4[0] = dV_dx4_R[0]

                                with hcl.if_(dV_dx5_L[0] > max_deriv5[0]):
                                    max_deriv5[0] = dV_dx5_L[0]
                                with hcl.if_(dV_dx5_R[0] > max_deriv5[0]):
                                    max_deriv5[0] = dV_dx5_R[0]

                                with hcl.if_(dV_dx6_L[0] > max_deriv6[0]):
                                    max_deriv6[0] = dV_dx6_L[0]
                                with hcl.if_(dV_dx6_R[0] > max_deriv6[0]):
                                    max_deriv6[0] = dV_dx6_R[0]

    # Calculate dissipation amount
    with hcl.Stage("Dissipation"):
        uOptL1 = hcl.scalar(0, "uOptL1")
        uOptL2 = hcl.scalar(0, "uOptL2")
        uOptL3 = hcl.scalar(0, "uOptL3")
        uOptL4 = hcl.scalar(0, "uOptL4")

        uOptU1 = hcl.scalar(0, "uOptU1")
        uOptU2 = hcl.scalar(0, "uOptU2")
        uOptU3 = hcl.scalar(0, "uOptU3")
        uOptU4 = hcl.scalar(0, "uOptU4")

        dOptL1 = hcl.scalar(0, "dOptL1")
        dOptL2 = hcl.scalar(0, "dOptL2")
        dOptL3 = hcl.scalar(0, "dOptL3")
        dOptL4 = hcl.scalar(0, "dOptL4")

        dOptU1 = hcl.scalar(0, "dOptU1")
        dOptU2 = hcl.scalar(0, "dOptU2")
        dOptU3 = hcl.scalar(0, "dOptU3")
        dOptU4 = hcl.scalar(0, "dOptU4")

        # Storing alphas
        alpha1 = hcl.scalar(0, "alpha1")
        alpha2 = hcl.scalar(0, "alpha2")
        alpha3 = hcl.scalar(0, "alpha3")
        alpha4 = hcl.scalar(0, "alpha4")
        alpha5 = hcl.scalar(0, "alpha5")
        alpha6 = hcl.scalar(0, "alpha6")

        # Find LOWER BOUND optimal disturbance
        dOptL = my_object.optDstb((min_deriv1[0], min_deriv2[0], min_deriv3[0], min_deriv4[0], min_deriv5[0], min_deriv6[0]))
        # Find UPPER BOUND optimal disturbance
        dOptU = my_object.optDstb((max_deriv1[0], max_deriv2[0], max_deriv3[0], max_deriv4[0], max_deriv5[0], max_deriv6[0]))
        with hcl.for_(0, V_init.shape[0], name="i") as i:
            with hcl.for_(0, V_init.shape[1], name="j") as j:
                with hcl.for_(0, V_init.shape[2], name="k") as k:
                    with hcl.for_(0, V_init.shape[3], name="l") as l:
                        with hcl.for_(0, V_init.shape[4], name="m") as m:
                            with hcl.for_(0, V_init.shape[5], name="n") as n:
                                dx_LL1 = hcl.scalar(0, "dx_LL1")
                                dx_LL2 = hcl.scalar(0, "dx_LL2")
                                dx_LL3 = hcl.scalar(0, "dx_LL3")
                                dx_LL4 = hcl.scalar(0, "dx_LL4")
                                dx_LL5 = hcl.scalar(0, "dx_LL5")
                                dx_LL6 = hcl.scalar(0, "dx_LL6")

                                dx_UL1 = hcl.scalar(0, "dx_UL1")
                                dx_UL2 = hcl.scalar(0, "dx_UL2")
                                dx_UL3 = hcl.scalar(0, "dx_UL3")
                                dx_UL4 = hcl.scalar(0, "dx_UL4")
                                dx_UL5 = hcl.scalar(0, "dx_UL5")
                                dx_UL6 = hcl.scalar(0, "dx_UL6")
                                #
                                dx_LU1 = hcl.scalar(0, "dx_LU1")
                                dx_LU2 = hcl.scalar(0, "dx_LU2")
                                dx_LU3 = hcl.scalar(0, "dx_LU3")
                                dx_LU4 = hcl.scalar(0, "dx_LU4")
                                dx_LU5 = hcl.scalar(0, "dx_LU5")
                                dx_LU6 = hcl.scalar(0, "dx_LU6")

                                dx_UU1 = hcl.scalar(0, "dx_UU1")
                                dx_UU2 = hcl.scalar(0, "dx_UU2")
                                dx_UU3 = hcl.scalar(0, "dx_UU3")
                                dx_UU4 = hcl.scalar(0, "dx_UU4")
                                dx_UU5 = hcl.scalar(0, "dx_UU5")
                                dx_UU6 = hcl.scalar(0, "dx_UU6")

                                # Find LOWER BOUND optimal control
                                uOptL = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), (min_deriv1[0], min_deriv2[0], min_deriv3[0], min_deriv4[0], min_deriv5[0], min_deriv6[0]))
                                # Find UPPER BOUND optimal control
                                uOptU = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), (max_deriv1[0], max_deriv2[0], max_deriv3[0], max_deriv4[0], max_deriv5[0], max_deriv6[0]))

                                # Get upper bound and lower bound rates of changes
                                dx_LL1[0], dx_LL2[0], dx_LL3[0], dx_LL4[0], dx_LL5[0], dx_LL6[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), uOptL, dOptL)
                                # Get absolute value of each
                                dx_LL1[0] = my_abs(dx_LL1[0])
                                dx_LL2[0] = my_abs(dx_LL2[0])
                                dx_LL3[0] = my_abs(dx_LL3[0])
                                dx_LL4[0] = my_abs(dx_LL4[0])
                                dx_LL5[0] = my_abs(dx_LL5[0])
                                dx_LL6[0] = my_abs(dx_LL6[0])

                                dx_UL1[0], dx_UL2[0], dx_UL3[0], dx_UL4[0], dx_UL5[0], dx_UL6[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), uOptU, dOptL)
                                # Get absolute value of each
                                dx_UL1[0] = my_abs(dx_UL1[0])
                                dx_UL2[0] = my_abs(dx_UL2[0])
                                dx_UL3[0] = my_abs(dx_UL3[0])
                                dx_UL4[0] = my_abs(dx_UL4[0])
                                dx_UL5[0] = my_abs(dx_UL5[0])
                                dx_UL6[0] = my_abs(dx_UL6[0])

                                # Set maximum alphas
                                alpha1[0] = my_max(dx_UL1[0], dx_LL1[0])
                                alpha2[0] = my_max(dx_UL2[0], dx_LL2[0])
                                alpha3[0] = my_max(dx_UL3[0], dx_LL3[0])
                                alpha4[0] = my_max(dx_UL4[0], dx_LL4[0])
                                alpha5[0] = my_max(dx_UL5[0], dx_LL5[0])
                                alpha6[0] = my_max(dx_UL6[0], dx_LL6[0])

                                dx_LU1[0], dx_LU2[0], dx_LU3[0], dx_LU4[0], dx_LU5[0], dx_LU6[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), uOptL, dOptU)
                                # Get absolute value of each
                                dx_LU1[0] = my_abs(dx_LU1[0])
                                dx_LU2[0] = my_abs(dx_LU2[0])
                                dx_LU3[0] = my_abs(dx_LU3[0])
                                dx_LU4[0] = my_abs(dx_LU4[0])
                                dx_LU5[0] = my_abs(dx_LU5[0])
                                dx_LU6[0] = my_abs(dx_LU6[0])

                                alpha1[0] = my_max(alpha1[0], dx_LU1[0])
                                alpha2[0] = my_max(alpha2[0], dx_LU2[0])
                                alpha3[0] = my_max(alpha3[0], dx_LU3[0])
                                alpha4[0] = my_max(alpha4[0], dx_LU4[0])
                                alpha5[0] = my_max(alpha5[0], dx_LU5[0])
                                alpha6[0] = my_max(alpha6[0], dx_LU6[0])

                                dx_UU1[0], dx_UU2[0], dx_UU3[0], dx_UU4[0], dx_UU5[0], dx_UU6[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), uOptU, dOptU)
                                dx_UU1[0] = my_abs(dx_UU1[0])
                                dx_UU2[0] = my_abs(dx_UU2[0])
                                dx_UU3[0] = my_abs(dx_UU3[0])
                                dx_UU4[0] = my_abs(dx_UU4[0])
                                dx_UU5[0] = my_abs(dx_UU5[0])
                                dx_UU6[0] = my_abs(dx_UU6[0])

                                alpha1[0] = my_max(alpha1[0], dx_UU1[0])
                                alpha2[0] = my_max(alpha2[0], dx_UU2[0])
                                alpha3[0] = my_max(alpha3[0], dx_UU3[0])
                                alpha4[0] = my_max(alpha4[0], dx_UU4[0])
                                alpha5[0] = my_max(alpha5[0], dx_UU5[0])
                                alpha6[0] = my_max(alpha6[0], dx_UU6[0])

                                diss = hcl.scalar(0, "diss")
                                diss[0] = 0.5*(deriv_diff1[i, j, k, l, m, n]*alpha1[0] + deriv_diff2[i, j, k, l, m, n]*alpha2[0] \
                                               + deriv_diff3[i, j, k, l, m, n]* alpha3[0] + deriv_diff4[i, j, k, l, m, n]* alpha4[0] \
                                               + deriv_diff5[i, j, k, l, m, n]* alpha5[0] + deriv_diff6[i, j, k, l, m, n]* alpha6[0])

                                # Finally
                                V_new[i, j, k, l, m, n] = -(V_new[i, j, k, l, m, n] - diss[0])
                                # Get maximum alphas in each dimension

                                # Calculate alphas
                                with hcl.if_(alpha1 > max_alpha1):
                                    max_alpha1[0] = alpha1[0]
                                with hcl.if_(alpha2 > max_alpha2):
                                    max_alpha2[0] = alpha2[0]
                                with hcl.if_(alpha3 > max_alpha3):
                                    max_alpha3[0] = alpha3[0]
                                with hcl.if_(alpha4 > max_alpha4):
                                    max_alpha4[0] = alpha4[0]
                                with hcl.if_(alpha5 > max_alpha5):
                                    max_alpha5[0] = alpha5[0]
                                with hcl.if_(alpha6 > max_alpha6):
                                    max_alpha6[0] = alpha6[0]


    # Determine time step
    delta_t = hcl.compute((1,), lambda x: step_bound(), name="delta_t")
    #hcl.update(t, lambda x: t[x] + delta_t[x])

    # Integrate
    #if compMethod == 'HJ_PDE':
    result = hcl.update(V_new, lambda i, j, k, l, m, n: V_init[i, j, k, l, m, n] + V_new[i, j, k, l, m, n] * delta_t[0])
    if compMethod == 'maxVWithV0':
        result = hcl.update(V_new, lambda i, j, k, l, m, n: maxVWithV0(i, j, k, l, m, n))
    if compMethod == 'maxVWithCStraint':
        result = hcl.update(V_new, lambda i, j, k, l, m, n: maxVWithCStraint(i, j, k, l, m, n))
    if compMethod == 'minVWithVInit':
        result = hcl.update(V_new, lambda i, j, k, l, m, n: minVWithVInit(i, j, k, l, m, n))
    # Copy V_new to V_init
    hcl.update(V_init, lambda i, j, k, l, m, n: V_new[i, j, k, l, m, n])
    return result

########################## 4D Graph definition #################################
def graph_4D(V_new, V_init, deriv_diff1, deriv_diff2, deriv_diff3, deriv_diff4, x1, x2, x3, x4, t, l0):
    # Maximum derivative for each dim
    max_deriv1 = hcl.scalar(-1e9, "max_deriv1")
    max_deriv2 = hcl.scalar(-1e9, "max_deriv2")
    max_deriv3 = hcl.scalar(-1e9, "max_deriv3")
    max_deriv4 = hcl.scalar(-1e9, "max_deriv4")

    # Min derivative for each dim
    min_deriv1 = hcl.scalar(1e9, "min_deriv1")
    min_deriv2 = hcl.scalar(1e9, "min_deriv2")
    min_deriv3 = hcl.scalar(1e9, "min_deriv3")
    min_deriv4 = hcl.scalar(1e9, "min_deriv4")

    # These variables are used to dissipation calculation
    max_alpha1 = hcl.scalar(-1e9, "max_alpha1")
    max_alpha2 = hcl.scalar(-1e9, "max_alpha2")
    max_alpha3 = hcl.scalar(-1e9, "max_alpha3")
    max_alpha4 = hcl.scalar(-1e9, "max_alpha4")

    def step_bound(): # Function to calculate time step
        stepBoundInv = hcl.scalar(0, "stepBoundInv")
        stepBound    = hcl.scalar(0, "stepBound")
        stepBoundInv[0] = max_alpha1[0]/g.dx[0] + max_alpha2[0]/g.dx[1] + max_alpha3[0]/g.dx[2] + max_alpha4[0]/g.dx[3]

        stepBound[0] = 0.8/stepBoundInv[0]
        with hcl.if_(stepBound > t[1] - t[0]):
            stepBound[0] = t[1] - t[0]

        # Update the lower time ranges
        t[0] = t[0] + stepBound[0]
        #t[0] = min_deriv2[0]
        return stepBound[0]

        # Min with V_before
    def minVWithVInit(i, j, k, l):
        with hcl.if_(V_new[i, j, k, l] > V_init[i, j, k, l]):
            V_new[i, j, k, l] = V_init[i, j, k, l]

    def maxVWithV0(i, j, k, l): # Take the max
        with hcl.if_(V_new[i, j, k, l] < l0[i, j, k, l]):
            V_new[i, j, k, l] = l0[i, j, k, l]

    # Calculate Hamiltonian for every grid point in V_init
    with hcl.Stage("Hamiltonian"):
        with hcl.for_(0, V_init.shape[0], name="i") as i:
            with hcl.for_(0, V_init.shape[1], name="j") as j:
                with hcl.for_(0, V_init.shape[2], name="k") as k:
                    with hcl.for_(0, V_init.shape[3], name="l") as l:
                        #Variables to calculate dV_dx
                        dV_dx1_L = hcl.scalar(0, "dV_dx1_L")
                        dV_dx1_R = hcl.scalar(0, "dV_dx1_R")
                        dV_dx1 = hcl.scalar(0, "dV_dx1")
                        dV_dx2_L = hcl.scalar(0, "dV_dx2_L")
                        dV_dx2_R = hcl.scalar(0, "dV_dx2_R")
                        dV_dx2 = hcl.scalar(0, "dV_dx2")
                        dV_dx3_L = hcl.scalar(0, "dV_dx3_L")
                        dV_dx3_R = hcl.scalar(0, "dV_dx3_R")
                        dV_dx3 = hcl.scalar(0, "dV_dx3")
                        dV_dx4_L = hcl.scalar(0, "dV_dx4_L")
                        dV_dx4_R = hcl.scalar(0, "dV_dx4_R")
                        dV_dx4 = hcl.scalar(0, "dV_dx4")

                        # No tensor slice operation
                        #dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
                        dV_dx1_L[0], dV_dx1_R[0] = spa_derivX1_4d(i, j, k, l, V_init, g)
                        dV_dx2_L[0], dV_dx2_R[0] = spa_derivX2_4d(i, j, k, l, V_init, g)
                        dV_dx3_L[0], dV_dx3_R[0] = spa_derivX3_4d(i, j, k, l, V_init, g)
                        dV_dx4_L[0], dV_dx4_R[0] = spa_derivX4_4d(i, j, k, l, V_init, g)

                        # Saves spatial derivative diff into tables
                        deriv_diff1[i, j, k, l] = dV_dx1_R[0] - dV_dx1_L[0]
                        deriv_diff2[i, j, k, l] = dV_dx2_R[0] - dV_dx2_L[0]
                        deriv_diff3[i, j, k, l] = dV_dx3_R[0] - dV_dx3_L[0]
                        deriv_diff4[i, j, k, l] = dV_dx4_R[0] - dV_dx4_L[0]

                        # Calculate average gradient
                        dV_dx1[0] = (dV_dx1_L + dV_dx1_R) / 2
                        dV_dx2[0] = (dV_dx2_L + dV_dx2_R) / 2
                        dV_dx3[0] = (dV_dx3_L + dV_dx3_R) / 2
                        dV_dx4[0] = (dV_dx4_L + dV_dx4_R) / 2

                        # Find optimal control
                        uOpt = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k], x4[l]), (dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0]))

                        # Find optimal disturbance
                        dOpt = my_object.optDstb((dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0]))
                        
                        # Find rates of changes based on dynamics equation
                        dx1_dt, dx2_dt, dx3_dt, dx4_dt = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]), uOpt, dOpt)

                        # Calculate Hamiltonian terms:
                        V_new[i, j, k, l] =  -(dx1_dt * dV_dx1[0] + dx2_dt * dV_dx2[0] + dx3_dt * dV_dx3[0] + dx4_dt * dV_dx4[0])

                        # Debugging
                        #V_new[i, j, k, l] = dV_dx2[0]

                        # Get derivMin
                        with hcl.if_(dV_dx1_L[0] < min_deriv1[0]):
                            min_deriv1[0] = dV_dx1_L[0]
                        with hcl.if_(dV_dx1_R[0] < min_deriv1[0]):
                            min_deriv1[0] = dV_dx1_R[0]

                        with hcl.if_(dV_dx2_L[0] < min_deriv2[0]):
                            min_deriv2[0] = dV_dx2_L[0]
                        with hcl.if_(dV_dx2_R[0] < min_deriv2[0]):
                            min_deriv2[0] = dV_dx2_R[0]

                        with hcl.if_(dV_dx3_L[0] < min_deriv3[0]):
                            min_deriv3[0] = dV_dx3_L[0]
                        with hcl.if_(dV_dx3_R[0] < min_deriv3[0]):
                            min_deriv3[0] = dV_dx3_R[0]

                        with hcl.if_(dV_dx4_L[0] < min_deriv4[0]):
                            min_deriv4[0] = dV_dx4_L[0]
                        with hcl.if_(dV_dx4_R[0] < min_deriv4[0]):
                            min_deriv4[0] = dV_dx4_R[0]

                        # Get derivMax
                        with hcl.if_(dV_dx1_L[0] > max_deriv1[0]):
                            max_deriv1[0] = dV_dx1_L[0]
                        with hcl.if_(dV_dx1_R[0] > max_deriv1[0]):
                            max_deriv1[0] = dV_dx1_R[0]

                        with hcl.if_(dV_dx2_L[0] > max_deriv2[0]):
                            max_deriv2[0] = dV_dx2_L[0]
                        with hcl.if_(dV_dx2_R[0] > max_deriv2[0]):
                            max_deriv2[0] = dV_dx2_R[0]

                        with hcl.if_(dV_dx3_L[0] > max_deriv3[0]):
                            max_deriv3[0] = dV_dx3_L[0]
                        with hcl.if_(dV_dx3_R[0] > max_deriv3[0]):
                            max_deriv3[0] = dV_dx3_R[0]

                        with hcl.if_(dV_dx4_L[0] > max_deriv4[0]):
                            max_deriv4[0] = dV_dx4_L[0]
                        with hcl.if_(dV_dx4_R[0] > max_deriv4[0]):
                            max_deriv4[0] = dV_dx4_R[0]

    # Calculate dissipation amount

    with hcl.Stage("Dissipation"):
         # Storing alphas
        alpha1 = hcl.scalar(0, "alpha1")
        alpha2 = hcl.scalar(0, "alpha2")
        alpha3 = hcl.scalar(0, "alpha3")
        alpha4 = hcl.scalar(0, "alpha4")

        with hcl.for_(0, V_init.shape[0], name="i") as i:
            with hcl.for_(0, V_init.shape[1], name="j") as j:
                with hcl.for_(0, V_init.shape[2], name="k") as k:
                    with hcl.for_(0, V_init.shape[3], name="l") as l:
                        dx_LL1 = hcl.scalar(0, "dx_LL1")
                        dx_LL2 = hcl.scalar(0, "dx_LL2")
                        dx_LL3 = hcl.scalar(0, "dx_LL3")
                        dx_LL4 = hcl.scalar(0, "dx_LL4")
                             
                        dx_UL1 = hcl.scalar(0, "dx_UL1")
                        dx_UL2 = hcl.scalar(0, "dx_UL2")
                        dx_UL3 = hcl.scalar(0, "dx_UL3")
                        dx_UL4 = hcl.scalar(0, "dx_UL4")

                        dx_LU1 = hcl.scalar(0, "dx_LU1")
                        dx_LU2 = hcl.scalar(0, "dx_LU2")
                        dx_LU3 = hcl.scalar(0, "dx_LU3")
                        dx_LU4 = hcl.scalar(0, "dx_LU4")

                        dx_UU1 = hcl.scalar(0, "dx_UU1")
                        dx_UU2 = hcl.scalar(0, "dx_UU2")
                        dx_UU3 = hcl.scalar(0, "dx_UU3")
                        dx_UU4 = hcl.scalar(0, "dx_UU4")
                              
                        # Find LOWER BOUND optimal control
                        uOptL = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k], x4[l]), (min_deriv1[0], min_deriv2[0], min_deriv3[0], min_deriv4[0]))
                        # Find UPPER BOUND optimal control
                        uOptU = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k], x4[l]), (max_deriv1[0], max_deriv2[0], max_deriv3[0], max_deriv4[0]))
                        # Find LOWER BOUND optimal disturbance
                        dOptL = my_object.optDstb((min_deriv1[0], min_deriv2[0], min_deriv3[0], min_deriv4[0]))
                        # Find UPPER BOUND optimal disturbance
                        dOptU = my_object.optDstb((max_deriv1[0], max_deriv2[0], max_deriv3[0], max_deriv4[0]))

                        # Find magnitude of rates of changes
                        dx_LL1[0], dx_LL2[0], dx_LL3[0], dx_LL4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]), uOptL, dOptL)
                        dx_LU1[0], dx_LU2[0], dx_LU3[0], dx_LU4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]), uOptL, dOptU)
                        dx_UL1[0], dx_UL2[0], dx_UL3[0], dx_UL4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]), uOptU, dOptL)
                        dx_UU1[0], dx_UU2[0], dx_UU3[0], dx_UU4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]), uOptU, dOptU)

                        # Calculate alpha
                        alpha1[0] = my_max(my_abs(dx_LL1[0]), my_abs(dx_LU1[0]))
                        alpha1[0] = my_max(alpha1[0], my_abs(dx_UL1[0]))
                        alpha1[0] = my_max(alpha1[0], my_abs(dx_UU1[0]))
                        
                        alpha2[0] = my_max(my_abs(dx_LL2[0]), my_abs(dx_LU2[0]))
                        alpha2[0] = my_max(alpha2[0], my_abs(dx_UL2[0]))
                        alpha2[0] = my_max(alpha2[0], my_abs(dx_UU2[0]))
                        
                        alpha3[0] = my_max(my_abs(dx_LL3[0]), my_abs(dx_LU3[0]))
                        alpha3[0] = my_max(alpha3[0], my_abs(dx_UL3[0]))
                        alpha3[0] = my_max(alpha3[0], my_abs(dx_UU3[0]))
                        
                        alpha4[0] = my_max(my_abs(dx_LL4[0]), my_abs(dx_LU4[0]))
                        alpha4[0] = my_max(alpha4[0], my_abs(dx_UL4[0]))
                        alpha4[0] = my_max(alpha4[0], my_abs(dx_UU4[0]))

                        diss = hcl.scalar(0, "diss")
                        diss[0] = 0.5*(deriv_diff1[i, j, k, l]*alpha1 + deriv_diff2[i, j, k, l]*alpha2 + deriv_diff3[i, j, k, l]* alpha3 + deriv_diff4[i, j, k, l]* alpha4)

                        # Finally
                        V_new[i, j, k, l] = -(V_new[i, j, k, l] - diss[0])
                        # Get maximum alphas in each dimension

                        # Calculate alphas
                        with hcl.if_(alpha1 > max_alpha1):
                            max_alpha1[0] = alpha1[0]
                        with hcl.if_(alpha2 > max_alpha2):
                            max_alpha2[0] = alpha2[0]
                        with hcl.if_(alpha3 > max_alpha3):
                            max_alpha3[0] = alpha3[0]
                        with hcl.if_(alpha4 > max_alpha4):
                            max_alpha4[0] = alpha4[0] 

    # Determine time step
    delta_t = hcl.compute((1,), lambda x: step_bound(), name="delta_t")
     # Integrate
    #if compMethod == 'HJ_PDE':
    result = hcl.update(V_new, lambda i, j, k, l: V_init[i, j, k, l] + V_new[i, j, k, l] * delta_t[0])
    if compMethod == 'maxVWithV0':
        result = hcl.update(V_new, lambda i, j, k, l: maxVWithV0(i, j, k, l))
    #if compMethod == 'maxVWithCStraint':
    #    result = hcl.update(V_new, lambda i, j, k, l: maxVWithCStraint(i, j, k, l))
    if compMethod == 'minVWithVInit':
        result = hcl.update(V_new, lambda i, j, k, l: minVWithVInit(i, j, k, l)) 
    # Copy V_new to V_init
    hcl.update(V_init, lambda i, j, k, l: V_new[i, j, k, l])
    return result
