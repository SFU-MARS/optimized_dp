import heterocl as hcl
import numpy as np
import time

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
def spa_derivX1_4d(i, j, k, l, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
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

def spa_derivX2_4d(i, j, k, l, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
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

def spa_derivX3_4d(i, j, k, l, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
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
        left_deriv[0] = (V[i, j - 1, k, l] - V[i, j, k, l]) / g.dx[1]
        right_deriv[0] = (right_boundary[0] - V[i, j, k, l]) / g.dx[1]
    with hcl.elif_(j != 0 and j != V.shape[1] - 1):
        left_deriv[0] = (V[i, j, k, l] - V[i, j - 1, k, l]) / g.dx[1]
        right_deriv[0] = (V[i, j + 1, k, l] - V[i, j, k, l]) / g.dx[1]
    return left_deriv[0], right_deriv[0]

def spa_derivX4_4d(i, j, k, l, V, g): # Left -> right == Outer Most -> Inner Most
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
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


############################## TODO: 6D DERIVATIVE FUNCTION #############################
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
