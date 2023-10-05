import heterocl as hcl
from odp.computeGraphs.CustomGraphFunctions import *

################## 1D SPATIAL DERIVATIVE FUNCTION #################
def spa_derivX(i, V, g):
    left_deriv = hcl.scalar(0, "left_deriv")
    right_deriv = hcl.scalar(0, "right_deriv")
    if 0 not in g.pDim:
        with hcl.if_(i == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[i] + my_abs(V[i + 1] - V[i]) * my_sign(\
                V[i])
            left_deriv[0] = (V[i] - left_boundary[0]) / g.dx[0]
            right_deriv[0] = (V[i + 1] - V[i]) / g.dx[0]
        with hcl.elif_(i == V.shape[0] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[i] + my_abs(V[i] - V[i - 1]) * my_sign(
                V[i])
            left_deriv[0] = (V[i] - V[i - 1]) / g.dx[0]
            right_deriv[0] = (right_boundary[0] - V[i]) / g.dx[0]
        with hcl.elif_(i != 0 and i != V.shape[0] - 1):
            left_deriv[0] = (V[i] - V[i - 1]) / g.dx[0]
            right_deriv[0] = (V[i + 1] - V[i]) / g.dx[0]
        return left_deriv[0], right_deriv[0]
    else:
        with hcl.if_(i == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V[V.shape[0] - 1]
            left_deriv[0] = (V[i] - left_boundary[0]) / g.dx[0]
            right_deriv[0] = (V[i + 1] - V[i]) / g.dx[0]
        with hcl.elif_(i == V.shape[0] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V[0]
            left_deriv[0] = (V[i] - V[i - 1]) / g.dx[0]
            right_deriv[0] = (right_boundary[0] - V[i]) / g.dx[0]
        with hcl.elif_(i != 0 and i != V.shape[0] - 1):
            left_deriv[0] = (V[i] - V[i - 1]) / g.dx[0]
            right_deriv[0] = (V[i + 1] - V[i]) / g.dx[0]
        return left_deriv[0], right_deriv[0]


