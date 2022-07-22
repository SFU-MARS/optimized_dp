import heterocl as hcl
import numpy as np
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

# ########################## 3D SPATIAL DERIVATIVE FUNCTION #################################
#
# def spa_derivX(i, j, k, V, g):
#     left_deriv = hcl.scalar(0, "left_deriv")
#     right_deriv = hcl.scalar(0, "right_deriv")
#     with hcl.if_(i == 0):
#         left_boundary = hcl.scalar(0, "left_boundary")
#         left_boundary[0] = V[k, j, i] + my_abs(V[k, j, i + 1] - V[k, j, i]) * my_sign(
#             V[k, j, i])
#         left_deriv[0] = (V[k, j, i] - left_boundary[0]) / g.dx[0]
#         right_deriv[0] = (V[k, j, i + 1] - V[k, j, i]) / g.dx[0]
#     with hcl.elif_(i == V.shape[2] - 1):
#         right_boundary = hcl.scalar(0, "right_boundary")
#         right_boundary[0] = V[k, j, i] + my_abs(V[k, j, i] - V[k, j, i - 1]) * my_sign(
#             V[k, j, i])
#         left_deriv[0] = (V[k, j, i] - V[k, j, i - 1]) / g.dx[0]
#         right_deriv[0] = (right_boundary[0] - V[k, j, i]) / g.dx[0]
#     with hcl.elif_(i != 0 and i != V.shape[2] - 1):
#         left_deriv[0] = (V[k, j, i] - V[k, j, i-1]) / g.dx[0]
#         right_deriv[0] = (V[k, j, i+1] - V[k, j, i]) / g.dx[0]
#     return left_deriv[0], right_deriv[0]
#
#
# def spa_derivY(i, j, k, V, g):
#     left_deriv = hcl.scalar(0, "left_deriv")
#     right_deriv = hcl.scalar(0, "right_deriv")
#     with hcl.if_(j == 0):
#         left_boundary = hcl.scalar(0, "left_boundary")
#         left_boundary[0] = V[k, j, i] + my_abs(V[k, j + 1, i] - V[k, j, i]) * my_sign(
#             V[k, j, i])
#         left_deriv[0] = (V[k, j, i] - left_boundary[0]) / g.dx[1]
#         right_deriv[0] = (V[k, j + 1, i] - V[k, j, i]) / g.dx[1]
#     with hcl.elif_(j == V.shape[1] - 1):
#         right_boundary = hcl.scalar(0, "right_boundary")
#         right_boundary[0] = V[k, j, i] + my_abs(V[k, j, i] - V[k, j - 1, i]) * my_sign(
#             V[k, j, i])
#         left_deriv[0] = (V[k, j, i] - V[k, j - 1, i]) / g.dx[1]
#         right_deriv[0] = (right_boundary[0] - V[k, j, i]) / g.dx[1]
#     with hcl.elif_(j != 0 and j != V.shape[1] - 1):
#         left_deriv[0] = (V[k, j, i] - V[k, j - 1, i]) / g.dx[1]
#         right_deriv[0] = (V[k, j + 1, i] - V[k, j, i]) / g.dx[1]
#     return left_deriv[0], right_deriv[0]
#
# def spa_derivT(i, j, k, V, g):
#     left_deriv = hcl.scalar(0, "left_deriv")
#     right_deriv = hcl.scalar(0, "right_deriv")
#     with hcl.if_(k == 0):
#         left_boundary = hcl.scalar(0, "left_boundary")
#         # left_boundary[0] = V_init[i,j,50]
#         left_boundary[0] = V[V.shape[0] - 1, j, i]
#         left_deriv[0] = (V[k, j, i] - left_boundary[0]) / g.dx[2]
#         right_deriv[0] = (V[k + 1, j, i] - V[k, j, i]) / g.dx[2]
#     with hcl.elif_(k == V.shape[0] - 1):
#         right_boundary = hcl.scalar(0, "right_boundary")
#         right_boundary[0] = V[0, j, i]
#         left_deriv[0] = (V[k, j, i] - V[k-1, j, i]) / g.dx[2]
#         right_deriv[0] = (right_boundary[0] - V[k, j, i]) / g.dx[2]
#     with hcl.elif_(k != 0 and k != V.shape[0] - 1):
#         left_deriv[0] = (V[k, j, i] - V[k-1, j, i]) / g.dx[2]
#         right_deriv[0] = (V[k+1, j, i] - V[k, j, i]) / g.dx[2]
#     return left_deriv[0], right_deriv[0]



