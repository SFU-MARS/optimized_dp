import heterocl as hcl 
from odp.computeGraphs.CustomGraphFunctions import *

def spa_derivX0_8d(i0, i1, i2, i3, i4, i5, i6, i7, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	if 0 not in g.pDim:
		with hcl.if_(i0 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0+1, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[0]
			right_deriv[0] = (V[i0+1, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[0]
		with hcl.elif_(i0 == V.shape[0] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0-1, i1, i2, i3, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0-1, i1, i2, i3, i4, i5, i6, i7]) / g.dx[0]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[0]
		with hcl.elif_(i0 != 0 and i0 != V.shape[0] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0-1, i1, i2, i3, i4, i5, i6, i7])/g.dx[0]
			right_deriv[0] = (V[i0+1, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[0]
		return left_deriv[0], right_deriv[0]
	else:
		with hcl.if_(i0 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[V.shape[0]-1, i1, i2, i3, i4, i5, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[0]
			right_deriv[0] = (V[i0+1, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[0]
		with hcl.elif_(i0 == V.shape[0] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[0, i1, i2, i3, i4, i5, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0-1, i1, i2, i3, i4, i5, i6, i7]) / g.dx[0]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[0]
		with hcl.elif_(i0 != 0 and i0 != V.shape[0] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0-1, i1, i2, i3, i4, i5, i6, i7])/g.dx[0]
			right_deriv[0] = (V[i0+1, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[0]
		return left_deriv[0], right_deriv[0]
def spa_derivX1_8d(i0, i1, i2, i3, i4, i5, i6, i7, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	if 1 not in g.pDim:
		with hcl.if_(i1 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1+1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[1]
			right_deriv[0] = (V[i0, i1+1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[1]
		with hcl.elif_(i1 == V.shape[1] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1-1, i2, i3, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1-1, i2, i3, i4, i5, i6, i7]) / g.dx[1]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[1]
		with hcl.elif_(i1 != 0 and i1 != V.shape[1] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1-1, i2, i3, i4, i5, i6, i7])/g.dx[1]
			right_deriv[0] = (V[i0, i1+1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[1]
		return left_deriv[0], right_deriv[0]
	else:
		with hcl.if_(i1 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, V.shape[1]-1, i2, i3, i4, i5, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[1]
			right_deriv[0] = (V[i0, i1+1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[1]
		with hcl.elif_(i1 == V.shape[1] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, 0, i2, i3, i4, i5, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1-1, i2, i3, i4, i5, i6, i7]) / g.dx[1]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[1]
		with hcl.elif_(i1 != 0 and i1 != V.shape[1] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1-1, i2, i3, i4, i5, i6, i7])/g.dx[1]
			right_deriv[0] = (V[i0, i1+1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[1]
		return left_deriv[0], right_deriv[0]
def spa_derivX2_8d(i0, i1, i2, i3, i4, i5, i6, i7, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	if 2 not in g.pDim:
		with hcl.if_(i2 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2+1, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[2]
			right_deriv[0] = (V[i0, i1, i2+1, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[2]
		with hcl.elif_(i2 == V.shape[2] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2-1, i3, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2-1, i3, i4, i5, i6, i7]) / g.dx[2]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[2]
		with hcl.elif_(i2 != 0 and i2 != V.shape[2] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2-1, i3, i4, i5, i6, i7])/g.dx[2]
			right_deriv[0] = (V[i0, i1, i2+1, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[2]
		return left_deriv[0], right_deriv[0]
	else:
		with hcl.if_(i2 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, V.shape[2]-1, i3, i4, i5, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[2]
			right_deriv[0] = (V[i0, i1, i2+1, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[2]
		with hcl.elif_(i2 == V.shape[2] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, 0, i3, i4, i5, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2-1, i3, i4, i5, i6, i7]) / g.dx[2]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[2]
		with hcl.elif_(i2 != 0 and i2 != V.shape[2] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2-1, i3, i4, i5, i6, i7])/g.dx[2]
			right_deriv[0] = (V[i0, i1, i2+1, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[2]
		return left_deriv[0], right_deriv[0]
def spa_derivX3_8d(i0, i1, i2, i3, i4, i5, i6, i7, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	if 3 not in g.pDim:
		with hcl.if_(i3 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3+1, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[3]
			right_deriv[0] = (V[i0, i1, i2, i3+1, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[3]
		with hcl.elif_(i3 == V.shape[3] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3-1, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3-1, i4, i5, i6, i7]) / g.dx[3]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[3]
		with hcl.elif_(i3 != 0 and i3 != V.shape[3] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3-1, i4, i5, i6, i7])/g.dx[3]
			right_deriv[0] = (V[i0, i1, i2, i3+1, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[3]
		return left_deriv[0], right_deriv[0]
	else:
		with hcl.if_(i3 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, V.shape[3]-1, i4, i5, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[3]
			right_deriv[0] = (V[i0, i1, i2, i3+1, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[3]
		with hcl.elif_(i3 == V.shape[3] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, 0, i4, i5, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3-1, i4, i5, i6, i7]) / g.dx[3]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[3]
		with hcl.elif_(i3 != 0 and i3 != V.shape[3] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3-1, i4, i5, i6, i7])/g.dx[3]
			right_deriv[0] = (V[i0, i1, i2, i3+1, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[3]
		return left_deriv[0], right_deriv[0]
def spa_derivX4_8d(i0, i1, i2, i3, i4, i5, i6, i7, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	if 4 not in g.pDim:
		with hcl.if_(i4 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4+1, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[4]
			right_deriv[0] = (V[i0, i1, i2, i3, i4+1, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[4]
		with hcl.elif_(i4 == V.shape[4] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4-1, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4-1, i5, i6, i7]) / g.dx[4]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[4]
		with hcl.elif_(i4 != 0 and i4 != V.shape[4] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4-1, i5, i6, i7])/g.dx[4]
			right_deriv[0] = (V[i0, i1, i2, i3, i4+1, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[4]
		return left_deriv[0], right_deriv[0]
	else:
		with hcl.if_(i4 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, V.shape[4]-1, i5, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[4]
			right_deriv[0] = (V[i0, i1, i2, i3, i4+1, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[4]
		with hcl.elif_(i4 == V.shape[4] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, 0, i5, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4-1, i5, i6, i7]) / g.dx[4]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[4]
		with hcl.elif_(i4 != 0 and i4 != V.shape[4] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4-1, i5, i6, i7])/g.dx[4]
			right_deriv[0] = (V[i0, i1, i2, i3, i4+1, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[4]
		return left_deriv[0], right_deriv[0]
def spa_derivX5_8d(i0, i1, i2, i3, i4, i5, i6, i7, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	if 5 not in g.pDim:
		with hcl.if_(i5 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4, i5+1, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[5]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5+1, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[5]
		with hcl.elif_(i5 == V.shape[5] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5-1, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5-1, i6, i7]) / g.dx[5]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[5]
		with hcl.elif_(i5 != 0 and i5 != V.shape[5] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5-1, i6, i7])/g.dx[5]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5+1, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[5]
		return left_deriv[0], right_deriv[0]
	else:
		with hcl.if_(i5 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, i4, V.shape[5]-1, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[5]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5+1, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[5]
		with hcl.elif_(i5 == V.shape[5] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, i4, 0, i6, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5-1, i6, i7]) / g.dx[5]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[5]
		with hcl.elif_(i5 != 0 and i5 != V.shape[5] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5-1, i6, i7])/g.dx[5]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5+1, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[5]
		return left_deriv[0], right_deriv[0]
def spa_derivX6_8d(i0, i1, i2, i3, i4, i5, i6, i7, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	if 6 not in g.pDim:
		with hcl.if_(i6 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4, i5, i6+1, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[6]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6+1, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[6]
		with hcl.elif_(i6 == V.shape[6] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6-1, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6-1, i7]) / g.dx[6]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[6]
		with hcl.elif_(i6 != 0 and i6 != V.shape[6] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6-1, i7])/g.dx[6]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6+1, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[6]
		return left_deriv[0], right_deriv[0]
	else:
		with hcl.if_(i6 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, i4, i5, V.shape[6]-1, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[6]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6+1, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[6]
		with hcl.elif_(i6 == V.shape[6] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, i4, i5, 0, i7]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6-1, i7]) / g.dx[6]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[6]
		with hcl.elif_(i6 != 0 and i6 != V.shape[6] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6-1, i7])/g.dx[6]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6+1, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[6]
		return left_deriv[0], right_deriv[0]
def spa_derivX7_8d(i0, i1, i2, i3, i4, i5, i6, i7, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	if 7 not in g.pDim:
		with hcl.if_(i7 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4, i5, i6, i7+1] - V[i0, i1, i2, i3, i4, i5, i6, i7]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[7]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7+1] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[7]
		with hcl.elif_(i7 == V.shape[7] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, i7] + my_abs(V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7-1]) * my_sign(V[i0, i1, i2, i3, i4, i5, i6, i7])
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7-1]) / g.dx[7]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[7]
		with hcl.elif_(i7 != 0 and i7 != V.shape[7] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7-1])/g.dx[7]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7+1] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[7]
		return left_deriv[0], right_deriv[0]
	else:
		with hcl.if_(i7 == 0):
			left_boundary = hcl.scalar(0, "left_boundary")
			left_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, V.shape[7]-1]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - left_boundary[0]) / g.dx[7]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7+1] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[7]
		with hcl.elif_(i7 == V.shape[7] - 1):
			right_boundary = hcl.scalar(0, "right_boundary")
			right_boundary[0] = V[i0, i1, i2, i3, i4, i5, i6, 0]
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7-1]) / g.dx[7]
			right_deriv[0] = (right_boundary[0] - V[i0, i1, i2, i3, i4, i5, i6, i7]) / g.dx[7]
		with hcl.elif_(i7 != 0 and i7 != V.shape[7] - 1):
			left_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7] - V[i0, i1, i2, i3, i4, i5, i6, i7-1])/g.dx[7]
			right_deriv[0] = (V[i0, i1, i2, i3, i4, i5, i6, i7+1] - V[i0, i1, i2, i3, i4, i5, i6, i7])/g.dx[7]
		return left_deriv[0], right_deriv[0]
