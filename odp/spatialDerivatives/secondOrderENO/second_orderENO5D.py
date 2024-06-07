import heterocl as hcl
from odp.computeGraphs.CustomGraphFunctions import *

def secondOrder_ENO5D_X0(i0, i1, i2, i3, i4, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	axis_step = g.dx[0]
	V_i_plus_1 = hcl.scalar(0, "V_i_plus_1")
	V_i_minus_1 = hcl.scalar(0, "V_i_minus_1")
	V_i_plus_2 = hcl.scalar(0, "V_i_plus_2")
	V_i_minus_2 = hcl.scalar(0, "V_i_minus_2")
	if 0 not in g.pDim:
		with hcl.if_(i0 == 0):
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0 + 1, i1, i2, i3, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4] + 2 * my_abs(V[i0 + 1, i1, i2, i3, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_1[0] = V[i0 + 1, i1, i2, i3, i4]
			V_i_plus_2[0] = V[i0 + 2, i1, i2, i3, i4]
		with hcl.elif_(i0 == 1):
			V_i_minus_1[0] = V[i0 - 1, i1, i2, i3, i4]
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4] - V[i0 - 1, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_1[0] = V[i0 + 1, i1, i2, i3, i4]
			V_i_plus_2[0] = V[i0 + 2, i1, i2, i3, i4]
		with hcl.elif_(i0 == V.shape[0] - 1):
			V_i_minus_1[0] = V[i0 - 1, i1, i2, i3, i4]
			V_i_minus_2[0] = V[i0 - 2, i1, i2, i3, i4]
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4] - V[i0 - 1, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_2[0] = V[i0, i1, i2, i3, i4] + 2 * my_abs(V[i0, i1, i2, i3, i4] - V[i0 - 1, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
		with hcl.elif_(i0 == V.shape[0] - 2):
			V_i_minus_1[0] = V[i0 - 1, i1, i2, i3, i4]
			V_i_minus_2[0] = V[i0 - 2, i1, i2, i3, i4]
			V_i_plus_1[0] = V[i0 + 1, i1, i2, i3, i4]
			V_i_plus_2[0] =V[i0, i1, i2, i3, i4] + my_abs(V[i0 + 1, i1, i2, i3, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
		with hcl.else_():
			V_i_minus_1[0] = V[i0 - 1, i1, i2, i3, i4]
			V_i_minus_2[0] = V[i0 - 2, i1, i2, i3, i4]
			V_i_plus_1[0] = V[i0 + 1, i1, i2, i3, i4]
			V_i_plus_2[0] = V[i0 + 2, i1, i2, i3, i4]
	else:
		with hcl.if_(i0 == 0):
			V_i_minus_1[0] = V[V.shape[0] - 1, i1, i2, i3, i4]
			V_i_minus_2[0] = V[V.shape[0] - 2, i1, i2, i3, i4]
			V_i_plus_1[0] = V[i0 + 1, i1, i2, i3, i4]
			V_i_plus_2[0] = V[i0 + 2, i1, i2, i3, i4]
		with hcl.elif_(i0 == 1):
			V_i_minus_1[0] = V[i0 - 1, i1, i2, i3, i4]
			V_i_minus_2[0] = V[V.shape[0] - 1, i1, i2, i3, i4]
			V_i_plus_1[0] = V[i0 + 1, i1, i2, i3, i4]
			V_i_plus_2[0] = V[i0 + 2, i1, i2, i3, i4]
		with hcl.elif_(i0 == V.shape[0] - 1):
			V_i_minus_1[0] = V[i0 - 1, i1, i2, i3, i4]
			V_i_minus_2[0] = V[i0 - 2, i1, i2, i3, i4]
			V_i_plus_1[0] = V[0, i1, i2, i3, i4]
			V_i_plus_2[0] = V[1, i1, i2, i3, i4]
		with hcl.elif_(i0 == V.shape[0] - 2):
			V_i_minus_1[0] = V[i0 - 1, i1, i2, i3, i4]
			V_i_minus_2[0] = V[i0 - 2, i1, i2, i3, i4]
			V_i_plus_1[0] = V[i0 + 1, i1, i2, i3, i4]
			V_i_plus_2[0] = V[0, i1, i2, i3, i4]
		with hcl.else_():
			V_i_minus_1[0] = V[i0 - 1, i1, i2, i3, i4]
			V_i_minus_2[0] = V[i0 - 2, i1, i2, i3, i4]
			V_i_plus_1[0] = V[i0 + 1, i1, i2, i3, i4]
			V_i_plus_2[0] = V[i0 + 2, i1, i2, i3, i4]
	D1_minus_2_plus_half = (V_i_minus_1[0] - V_i_minus_2[0]) / axis_step
	D1_minus_1_plus_half = (V[i0, i1, i2, i3, i4] - V_i_minus_1[0]) / axis_step
	D1_0_plus_half = (V_i_plus_1[0] - V[i0, i1, i2, i3, i4]) / axis_step
	D1_plus_1_plus_half = (V_i_plus_2[0] - V_i_plus_1[0]) / axis_step

	D2_minus_1 = (D1_minus_1_plus_half - D1_minus_2_plus_half) / (2 * axis_step)
	D2_0 = (D1_0_plus_half - D1_minus_1_plus_half) / (2 * axis_step)
	D2_plus_1 = (D1_plus_1_plus_half - D1_0_plus_half) / (2 * axis_step)

	with hcl.if_(my_abs(D2_minus_1) <= my_abs(D2_0)):
		left_deriv[0] = D1_minus_1_plus_half + D2_minus_1 * axis_step
	with hcl.else_():
		left_deriv[0] = D1_minus_1_plus_half + D2_0 * axis_step

	with hcl.if_(my_abs(D2_0) <= my_abs(D2_plus_1)):
		right_deriv[0] = D1_0_plus_half - D2_0 * axis_step
	with hcl.else_():
		right_deriv[0] = D1_0_plus_half - D2_plus_1 * axis_step
	return left_deriv[0], right_deriv[0]

def secondOrder_ENO5D_X1(i0, i1, i2, i3, i4, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	axis_step = g.dx[1]
	V_i_plus_1 = hcl.scalar(0, "V_i_plus_1")
	V_i_minus_1 = hcl.scalar(0, "V_i_minus_1")
	V_i_plus_2 = hcl.scalar(0, "V_i_plus_2")
	V_i_minus_2 = hcl.scalar(0, "V_i_minus_2")
	if 1 not in g.pDim:
		with hcl.if_(i1 == 0):
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1 + 1, i2, i3, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4] + 2 * my_abs(V[i0, i1 + 1, i2, i3, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_1[0] = V[i0, i1 + 1, i2, i3, i4]
			V_i_plus_2[0] = V[i0, i1 + 2, i2, i3, i4]
		with hcl.elif_(i1 == 1):
			V_i_minus_1[0] = V[i0, i1 - 1, i2, i3, i4]
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1 - 1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_1[0] = V[i0, i1 + 1, i2, i3, i4]
			V_i_plus_2[0] = V[i0, i1 + 2, i2, i3, i4]
		with hcl.elif_(i1 == V.shape[1] - 1):
			V_i_minus_1[0] = V[i0, i1 - 1, i2, i3, i4]
			V_i_minus_2[0] = V[i0, i1 - 2, i2, i3, i4]
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1 - 1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_2[0] = V[i0, i1, i2, i3, i4] + 2 * my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1 - 1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
		with hcl.elif_(i1 == V.shape[1] - 2):
			V_i_minus_1[0] = V[i0, i1 - 1, i2, i3, i4]
			V_i_minus_2[0] = V[i0, i1 - 2, i2, i3, i4]
			V_i_plus_1[0] = V[i0, i1 + 1, i2, i3, i4]
			V_i_plus_2[0] =V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1 + 1, i2, i3, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
		with hcl.else_():
			V_i_minus_1[0] = V[i0, i1 - 1, i2, i3, i4]
			V_i_minus_2[0] = V[i0, i1 - 2, i2, i3, i4]
			V_i_plus_1[0] = V[i0, i1 + 1, i2, i3, i4]
			V_i_plus_2[0] = V[i0, i1 + 2, i2, i3, i4]
	else:
		with hcl.if_(i1 == 0):
			V_i_minus_1[0] = V[i0, V.shape[1] - 1, i2, i3, i4]
			V_i_minus_2[0] = V[i0, V.shape[1] - 2, i2, i3, i4]
			V_i_plus_1[0] = V[i0, i1 + 1, i2, i3, i4]
			V_i_plus_2[0] = V[i0, i1 + 2, i2, i3, i4]
		with hcl.elif_(i1 == 1):
			V_i_minus_1[0] = V[i0, i1 - 1, i2, i3, i4]
			V_i_minus_2[0] = V[i0, V.shape[1] - 1, i2, i3, i4]
			V_i_plus_1[0] = V[i0, i1 + 1, i2, i3, i4]
			V_i_plus_2[0] = V[i0, i1 + 2, i2, i3, i4]
		with hcl.elif_(i1 == V.shape[1] - 1):
			V_i_minus_1[0] = V[i0, i1 - 1, i2, i3, i4]
			V_i_minus_2[0] = V[i0, i1 - 2, i2, i3, i4]
			V_i_plus_1[0] = V[i0, 0, i2, i3, i4]
			V_i_plus_2[0] = V[i0, 1, i2, i3, i4]
		with hcl.elif_(i1 == V.shape[1] - 2):
			V_i_minus_1[0] = V[i0, i1 - 1, i2, i3, i4]
			V_i_minus_2[0] = V[i0, i1 - 2, i2, i3, i4]
			V_i_plus_1[0] = V[i0, i1 + 1, i2, i3, i4]
			V_i_plus_2[0] = V[i0, 0, i2, i3, i4]
		with hcl.else_():
			V_i_minus_1[0] = V[i0, i1 - 1, i2, i3, i4]
			V_i_minus_2[0] = V[i0, i1 - 2, i2, i3, i4]
			V_i_plus_1[0] = V[i0, i1 + 1, i2, i3, i4]
			V_i_plus_2[0] = V[i0, i1 + 2, i2, i3, i4]
	D1_minus_2_plus_half = (V_i_minus_1[0] - V_i_minus_2[0]) / axis_step
	D1_minus_1_plus_half = (V[i0, i1, i2, i3, i4] - V_i_minus_1[0]) / axis_step
	D1_0_plus_half = (V_i_plus_1[0] - V[i0, i1, i2, i3, i4]) / axis_step
	D1_plus_1_plus_half = (V_i_plus_2[0] - V_i_plus_1[0]) / axis_step

	D2_minus_1 = (D1_minus_1_plus_half - D1_minus_2_plus_half) / (2 * axis_step)
	D2_0 = (D1_0_plus_half - D1_minus_1_plus_half) / (2 * axis_step)
	D2_plus_1 = (D1_plus_1_plus_half - D1_0_plus_half) / (2 * axis_step)

	with hcl.if_(my_abs(D2_minus_1) <= my_abs(D2_0)):
		left_deriv[0] = D1_minus_1_plus_half + D2_minus_1 * axis_step
	with hcl.else_():
		left_deriv[0] = D1_minus_1_plus_half + D2_0 * axis_step

	with hcl.if_(my_abs(D2_0) <= my_abs(D2_plus_1)):
		right_deriv[0] = D1_0_plus_half - D2_0 * axis_step
	with hcl.else_():
		right_deriv[0] = D1_0_plus_half - D2_plus_1 * axis_step
	return left_deriv[0], right_deriv[0]

def secondOrder_ENO5D_X2(i0, i1, i2, i3, i4, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	axis_step = g.dx[2]
	V_i_plus_1 = hcl.scalar(0, "V_i_plus_1")
	V_i_minus_1 = hcl.scalar(0, "V_i_minus_1")
	V_i_plus_2 = hcl.scalar(0, "V_i_plus_2")
	V_i_minus_2 = hcl.scalar(0, "V_i_minus_2")
	if 2 not in g.pDim:
		with hcl.if_(i2 == 0):
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2 + 1, i3, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4] + 2 * my_abs(V[i0, i1, i2 + 1, i3, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_1[0] = V[i0, i1, i2 + 1, i3, i4]
			V_i_plus_2[0] = V[i0, i1, i2 + 2, i3, i4]
		with hcl.elif_(i2 == 1):
			V_i_minus_1[0] = V[i0, i1, i2 - 1, i3, i4]
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1, i2 - 1, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_1[0] = V[i0, i1, i2 + 1, i3, i4]
			V_i_plus_2[0] = V[i0, i1, i2 + 2, i3, i4]
		with hcl.elif_(i2 == V.shape[2] - 1):
			V_i_minus_1[0] = V[i0, i1, i2 - 1, i3, i4]
			V_i_minus_2[0] = V[i0, i1, i2 - 2, i3, i4]
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1, i2 - 1, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_2[0] = V[i0, i1, i2, i3, i4] + 2 * my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1, i2 - 1, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
		with hcl.elif_(i2 == V.shape[2] - 2):
			V_i_minus_1[0] = V[i0, i1, i2 - 1, i3, i4]
			V_i_minus_2[0] = V[i0, i1, i2 - 2, i3, i4]
			V_i_plus_1[0] = V[i0, i1, i2 + 1, i3, i4]
			V_i_plus_2[0] =V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2 + 1, i3, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
		with hcl.else_():
			V_i_minus_1[0] = V[i0, i1, i2 - 1, i3, i4]
			V_i_minus_2[0] = V[i0, i1, i2 - 2, i3, i4]
			V_i_plus_1[0] = V[i0, i1, i2 + 1, i3, i4]
			V_i_plus_2[0] = V[i0, i1, i2 + 2, i3, i4]
	else:
		with hcl.if_(i2 == 0):
			V_i_minus_1[0] = V[i0, i1, V.shape[2] - 1, i3, i4]
			V_i_minus_2[0] = V[i0, i1, V.shape[2] - 2, i3, i4]
			V_i_plus_1[0] = V[i0, i1, i2 + 1, i3, i4]
			V_i_plus_2[0] = V[i0, i1, i2 + 2, i3, i4]
		with hcl.elif_(i2 == 1):
			V_i_minus_1[0] = V[i0, i1, i2 - 1, i3, i4]
			V_i_minus_2[0] = V[i0, i1, V.shape[2] - 1, i3, i4]
			V_i_plus_1[0] = V[i0, i1, i2 + 1, i3, i4]
			V_i_plus_2[0] = V[i0, i1, i2 + 2, i3, i4]
		with hcl.elif_(i2 == V.shape[2] - 1):
			V_i_minus_1[0] = V[i0, i1, i2 - 1, i3, i4]
			V_i_minus_2[0] = V[i0, i1, i2 - 2, i3, i4]
			V_i_plus_1[0] = V[i0, i1, 0, i3, i4]
			V_i_plus_2[0] = V[i0, i1, 1, i3, i4]
		with hcl.elif_(i2 == V.shape[2] - 2):
			V_i_minus_1[0] = V[i0, i1, i2 - 1, i3, i4]
			V_i_minus_2[0] = V[i0, i1, i2 - 2, i3, i4]
			V_i_plus_1[0] = V[i0, i1, i2 + 1, i3, i4]
			V_i_plus_2[0] = V[i0, i1, 0, i3, i4]
		with hcl.else_():
			V_i_minus_1[0] = V[i0, i1, i2 - 1, i3, i4]
			V_i_minus_2[0] = V[i0, i1, i2 - 2, i3, i4]
			V_i_plus_1[0] = V[i0, i1, i2 + 1, i3, i4]
			V_i_plus_2[0] = V[i0, i1, i2 + 2, i3, i4]
	D1_minus_2_plus_half = (V_i_minus_1[0] - V_i_minus_2[0]) / axis_step
	D1_minus_1_plus_half = (V[i0, i1, i2, i3, i4] - V_i_minus_1[0]) / axis_step
	D1_0_plus_half = (V_i_plus_1[0] - V[i0, i1, i2, i3, i4]) / axis_step
	D1_plus_1_plus_half = (V_i_plus_2[0] - V_i_plus_1[0]) / axis_step

	D2_minus_1 = (D1_minus_1_plus_half - D1_minus_2_plus_half) / (2 * axis_step)
	D2_0 = (D1_0_plus_half - D1_minus_1_plus_half) / (2 * axis_step)
	D2_plus_1 = (D1_plus_1_plus_half - D1_0_plus_half) / (2 * axis_step)

	with hcl.if_(my_abs(D2_minus_1) <= my_abs(D2_0)):
		left_deriv[0] = D1_minus_1_plus_half + D2_minus_1 * axis_step
	with hcl.else_():
		left_deriv[0] = D1_minus_1_plus_half + D2_0 * axis_step

	with hcl.if_(my_abs(D2_0) <= my_abs(D2_plus_1)):
		right_deriv[0] = D1_0_plus_half - D2_0 * axis_step
	with hcl.else_():
		right_deriv[0] = D1_0_plus_half - D2_plus_1 * axis_step
	return left_deriv[0], right_deriv[0]

def secondOrder_ENO5D_X3(i0, i1, i2, i3, i4, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	axis_step = g.dx[3]
	V_i_plus_1 = hcl.scalar(0, "V_i_plus_1")
	V_i_minus_1 = hcl.scalar(0, "V_i_minus_1")
	V_i_plus_2 = hcl.scalar(0, "V_i_plus_2")
	V_i_minus_2 = hcl.scalar(0, "V_i_minus_2")
	if 3 not in g.pDim:
		with hcl.if_(i3 == 0):
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3 + 1, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4] + 2 * my_abs(V[i0, i1, i2, i3 + 1, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_1[0] = V[i0, i1, i2, i3 + 1, i4]
			V_i_plus_2[0] = V[i0, i1, i2, i3 + 2, i4]
		with hcl.elif_(i3 == 1):
			V_i_minus_1[0] = V[i0, i1, i2, i3 - 1, i4]
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1, i2, i3 - 1, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_1[0] = V[i0, i1, i2, i3 + 1, i4]
			V_i_plus_2[0] = V[i0, i1, i2, i3 + 2, i4]
		with hcl.elif_(i3 == V.shape[3] - 1):
			V_i_minus_1[0] = V[i0, i1, i2, i3 - 1, i4]
			V_i_minus_2[0] = V[i0, i1, i2, i3 - 2, i4]
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1, i2, i3 - 1, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_2[0] = V[i0, i1, i2, i3, i4] + 2 * my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1, i2, i3 - 1, i4]) * my_sign(V[i0, i1, i2, i3, i4])
		with hcl.elif_(i3 == V.shape[3] - 2):
			V_i_minus_1[0] = V[i0, i1, i2, i3 - 1, i4]
			V_i_minus_2[0] = V[i0, i1, i2, i3 - 2, i4]
			V_i_plus_1[0] = V[i0, i1, i2, i3 + 1, i4]
			V_i_plus_2[0] =V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3 + 1, i4] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
		with hcl.else_():
			V_i_minus_1[0] = V[i0, i1, i2, i3 - 1, i4]
			V_i_minus_2[0] = V[i0, i1, i2, i3 - 2, i4]
			V_i_plus_1[0] = V[i0, i1, i2, i3 + 1, i4]
			V_i_plus_2[0] = V[i0, i1, i2, i3 + 2, i4]
	else:
		with hcl.if_(i3 == 0):
			V_i_minus_1[0] = V[i0, i1, i2, V.shape[3] - 1, i4]
			V_i_minus_2[0] = V[i0, i1, i2, V.shape[3] - 2, i4]
			V_i_plus_1[0] = V[i0, i1, i2, i3 + 1, i4]
			V_i_plus_2[0] = V[i0, i1, i2, i3 + 2, i4]
		with hcl.elif_(i3 == 1):
			V_i_minus_1[0] = V[i0, i1, i2, i3 - 1, i4]
			V_i_minus_2[0] = V[i0, i1, i2, V.shape[3] - 1, i4]
			V_i_plus_1[0] = V[i0, i1, i2, i3 + 1, i4]
			V_i_plus_2[0] = V[i0, i1, i2, i3 + 2, i4]
		with hcl.elif_(i3 == V.shape[3] - 1):
			V_i_minus_1[0] = V[i0, i1, i2, i3 - 1, i4]
			V_i_minus_2[0] = V[i0, i1, i2, i3 - 2, i4]
			V_i_plus_1[0] = V[i0, i1, i2, 0, i4]
			V_i_plus_2[0] = V[i0, i1, i2, 1, i4]
		with hcl.elif_(i3 == V.shape[3] - 2):
			V_i_minus_1[0] = V[i0, i1, i2, i3 - 1, i4]
			V_i_minus_2[0] = V[i0, i1, i2, i3 - 2, i4]
			V_i_plus_1[0] = V[i0, i1, i2, i3 + 1, i4]
			V_i_plus_2[0] = V[i0, i1, i2, 0, i4]
		with hcl.else_():
			V_i_minus_1[0] = V[i0, i1, i2, i3 - 1, i4]
			V_i_minus_2[0] = V[i0, i1, i2, i3 - 2, i4]
			V_i_plus_1[0] = V[i0, i1, i2, i3 + 1, i4]
			V_i_plus_2[0] = V[i0, i1, i2, i3 + 2, i4]
	D1_minus_2_plus_half = (V_i_minus_1[0] - V_i_minus_2[0]) / axis_step
	D1_minus_1_plus_half = (V[i0, i1, i2, i3, i4] - V_i_minus_1[0]) / axis_step
	D1_0_plus_half = (V_i_plus_1[0] - V[i0, i1, i2, i3, i4]) / axis_step
	D1_plus_1_plus_half = (V_i_plus_2[0] - V_i_plus_1[0]) / axis_step

	D2_minus_1 = (D1_minus_1_plus_half - D1_minus_2_plus_half) / (2 * axis_step)
	D2_0 = (D1_0_plus_half - D1_minus_1_plus_half) / (2 * axis_step)
	D2_plus_1 = (D1_plus_1_plus_half - D1_0_plus_half) / (2 * axis_step)

	with hcl.if_(my_abs(D2_minus_1) <= my_abs(D2_0)):
		left_deriv[0] = D1_minus_1_plus_half + D2_minus_1 * axis_step
	with hcl.else_():
		left_deriv[0] = D1_minus_1_plus_half + D2_0 * axis_step

	with hcl.if_(my_abs(D2_0) <= my_abs(D2_plus_1)):
		right_deriv[0] = D1_0_plus_half - D2_0 * axis_step
	with hcl.else_():
		right_deriv[0] = D1_0_plus_half - D2_plus_1 * axis_step
	return left_deriv[0], right_deriv[0]

def secondOrder_ENO5D_X4(i0, i1, i2, i3, i4, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	axis_step = g.dx[4]
	V_i_plus_1 = hcl.scalar(0, "V_i_plus_1")
	V_i_minus_1 = hcl.scalar(0, "V_i_minus_1")
	V_i_plus_2 = hcl.scalar(0, "V_i_plus_2")
	V_i_minus_2 = hcl.scalar(0, "V_i_minus_2")
	if 4 not in g.pDim:
		with hcl.if_(i4 == 0):
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4 + 1] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4] + 2 * my_abs(V[i0, i1, i2, i3, i4 + 1] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4 + 1]
			V_i_plus_2[0] = V[i0, i1, i2, i3, i4 + 2]
		with hcl.elif_(i4 == 1):
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4 - 1]
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1, i2, i3, i4 - 1]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4 + 1]
			V_i_plus_2[0] = V[i0, i1, i2, i3, i4 + 2]
		with hcl.elif_(i4 == V.shape[4] - 1):
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4 - 1]
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4 - 2]
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1, i2, i3, i4 - 1]) * my_sign(V[i0, i1, i2, i3, i4])
			V_i_plus_2[0] = V[i0, i1, i2, i3, i4] + 2 * my_abs(V[i0, i1, i2, i3, i4] - V[i0, i1, i2, i3, i4 - 1]) * my_sign(V[i0, i1, i2, i3, i4])
		with hcl.elif_(i4 == V.shape[4] - 2):
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4 - 1]
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4 - 2]
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4 + 1]
			V_i_plus_2[0] =V[i0, i1, i2, i3, i4] + my_abs(V[i0, i1, i2, i3, i4 + 1] - V[i0, i1, i2, i3, i4]) * my_sign(V[i0, i1, i2, i3, i4])
		with hcl.else_():
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4 - 1]
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4 - 2]
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4 + 1]
			V_i_plus_2[0] = V[i0, i1, i2, i3, i4 + 2]
	else:
		with hcl.if_(i4 == 0):
			V_i_minus_1[0] = V[i0, i1, i2, i3, V.shape[4] - 1]
			V_i_minus_2[0] = V[i0, i1, i2, i3, V.shape[4] - 2]
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4 + 1]
			V_i_plus_2[0] = V[i0, i1, i2, i3, i4 + 2]
		with hcl.elif_(i4 == 1):
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4 - 1]
			V_i_minus_2[0] = V[i0, i1, i2, i3, V.shape[4] - 1]
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4 + 1]
			V_i_plus_2[0] = V[i0, i1, i2, i3, i4 + 2]
		with hcl.elif_(i4 == V.shape[4] - 1):
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4 - 1]
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4 - 2]
			V_i_plus_1[0] = V[i0, i1, i2, i3, 0]
			V_i_plus_2[0] = V[i0, i1, i2, i3, 1]
		with hcl.elif_(i4 == V.shape[4] - 2):
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4 - 1]
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4 - 2]
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4 + 1]
			V_i_plus_2[0] = V[i0, i1, i2, i3, 0]
		with hcl.else_():
			V_i_minus_1[0] = V[i0, i1, i2, i3, i4 - 1]
			V_i_minus_2[0] = V[i0, i1, i2, i3, i4 - 2]
			V_i_plus_1[0] = V[i0, i1, i2, i3, i4 + 1]
			V_i_plus_2[0] = V[i0, i1, i2, i3, i4 + 2]
	D1_minus_2_plus_half = (V_i_minus_1[0] - V_i_minus_2[0]) / axis_step
	D1_minus_1_plus_half = (V[i0, i1, i2, i3, i4] - V_i_minus_1[0]) / axis_step
	D1_0_plus_half = (V_i_plus_1[0] - V[i0, i1, i2, i3, i4]) / axis_step
	D1_plus_1_plus_half = (V_i_plus_2[0] - V_i_plus_1[0]) / axis_step

	D2_minus_1 = (D1_minus_1_plus_half - D1_minus_2_plus_half) / (2 * axis_step)
	D2_0 = (D1_0_plus_half - D1_minus_1_plus_half) / (2 * axis_step)
	D2_plus_1 = (D1_plus_1_plus_half - D1_0_plus_half) / (2 * axis_step)

	with hcl.if_(my_abs(D2_minus_1) <= my_abs(D2_0)):
		left_deriv[0] = D1_minus_1_plus_half + D2_minus_1 * axis_step
	with hcl.else_():
		left_deriv[0] = D1_minus_1_plus_half + D2_0 * axis_step

	with hcl.if_(my_abs(D2_0) <= my_abs(D2_plus_1)):
		right_deriv[0] = D1_0_plus_half - D2_0 * axis_step
	with hcl.else_():
		right_deriv[0] = D1_0_plus_half - D2_plus_1 * axis_step
	return left_deriv[0], right_deriv[0]

