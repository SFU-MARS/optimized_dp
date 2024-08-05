import heterocl as hcl
from odp.computeGraphs.CustomGraphFunctions import *

def secondOrder_ENO2D_X0(i0, i1, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	axis_step = g.dx[0]
	V_i_plus_1 = hcl.scalar(0, "V_i_plus_1")
	V_i_minus_1 = hcl.scalar(0, "V_i_minus_1")
	V_i_plus_2 = hcl.scalar(0, "V_i_plus_2")
	V_i_minus_2 = hcl.scalar(0, "V_i_minus_2")
	if 0 not in g.pDim:
		with hcl.if_(i0 == 0):
			V_i_minus_1[0] = V[i0, i1] + my_abs(V[i0 + 1, i1] - V[i0, i1]) * my_sign(V[i0, i1])
			V_i_minus_2[0] = V[i0, i1] + 2 * my_abs(V[i0 + 1, i1] - V[i0, i1]) * my_sign(V[i0, i1])
			V_i_plus_1[0] = V[i0 + 1, i1]
			V_i_plus_2[0] = V[i0 + 2, i1]
		with hcl.elif_(i0 == 1):
			V_i_minus_1[0] = V[i0 - 1, i1]
			V_i_minus_2[0] = V[i0, i1] + my_abs(V[i0, i1] - V[i0 - 1, i1]) * my_sign(V[i0, i1])
			V_i_plus_1[0] = V[i0 + 1, i1]
			V_i_plus_2[0] = V[i0 + 2, i1]
		with hcl.elif_(i0 == V.shape[0] - 1):
			V_i_minus_1[0] = V[i0 - 1, i1]
			V_i_minus_2[0] = V[i0 - 2, i1]
			V_i_plus_1[0] = V[i0, i1] + my_abs(V[i0, i1] - V[i0 - 1, i1]) * my_sign(V[i0, i1])
			V_i_plus_2[0] = V[i0, i1] + 2 * my_abs(V[i0, i1] - V[i0 - 1, i1]) * my_sign(V[i0, i1])
		with hcl.elif_(i0 == V.shape[0] - 2):
			V_i_minus_1[0] = V[i0 - 1, i1]
			V_i_minus_2[0] = V[i0 - 2, i1]
			V_i_plus_1[0] = V[i0 + 1, i1]
			V_i_plus_2[0] =V[i0, i1] + my_abs(V[i0 + 1, i1] - V[i0, i1]) * my_sign(V[i0, i1])
		with hcl.else_():
			V_i_minus_1[0] = V[i0 - 1, i1]
			V_i_minus_2[0] = V[i0 - 2, i1]
			V_i_plus_1[0] = V[i0 + 1, i1]
			V_i_plus_2[0] = V[i0 + 2, i1]
	else:
		with hcl.if_(i0 == 0):
			V_i_minus_1[0] = V[V.shape[0] - 1, i1]
			V_i_minus_2[0] = V[V.shape[0] - 2, i1]
			V_i_plus_1[0] = V[i0 + 1, i1]
			V_i_plus_2[0] = V[i0 + 2, i1]
		with hcl.elif_(i0 == 1):
			V_i_minus_1[0] = V[i0 - 1, i1]
			V_i_minus_2[0] = V[V.shape[0] - 1, i1]
			V_i_plus_1[0] = V[i0 + 1, i1]
			V_i_plus_2[0] = V[i0 + 2, i1]
		with hcl.elif_(i0 == V.shape[0] - 1):
			V_i_minus_1[0] = V[i0 - 1, i1]
			V_i_minus_2[0] = V[i0 - 2, i1]
			V_i_plus_1[0] = V[0, i1]
			V_i_plus_2[0] = V[1, i1]
		with hcl.elif_(i0 == V.shape[0] - 2):
			V_i_minus_1[0] = V[i0 - 1, i1]
			V_i_minus_2[0] = V[i0 - 2, i1]
			V_i_plus_1[0] = V[i0 + 1, i1]
			V_i_plus_2[0] = V[0, i1]
		with hcl.else_():
			V_i_minus_1[0] = V[i0 - 1, i1]
			V_i_minus_2[0] = V[i0 - 2, i1]
			V_i_plus_1[0] = V[i0 + 1, i1]
			V_i_plus_2[0] = V[i0 + 2, i1]
	D1_minus_2_plus_half = (V_i_minus_1[0] - V_i_minus_2[0]) / axis_step
	D1_minus_1_plus_half = (V[i0, i1] - V_i_minus_1[0]) / axis_step
	D1_0_plus_half = (V_i_plus_1[0] - V[i0, i1]) / axis_step
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

def secondOrder_ENO2D_X1(i0, i1, V, g):
	left_deriv = hcl.scalar(0, "left_deriv")
	right_deriv = hcl.scalar(0, "right_deriv")
	axis_step = g.dx[1]
	V_i_plus_1 = hcl.scalar(0, "V_i_plus_1")
	V_i_minus_1 = hcl.scalar(0, "V_i_minus_1")
	V_i_plus_2 = hcl.scalar(0, "V_i_plus_2")
	V_i_minus_2 = hcl.scalar(0, "V_i_minus_2")
	if 1 not in g.pDim:
		with hcl.if_(i1 == 0):
			V_i_minus_1[0] = V[i0, i1] + my_abs(V[i0, i1 + 1] - V[i0, i1]) * my_sign(V[i0, i1])
			V_i_minus_2[0] = V[i0, i1] + 2 * my_abs(V[i0, i1 + 1] - V[i0, i1]) * my_sign(V[i0, i1])
			V_i_plus_1[0] = V[i0, i1 + 1]
			V_i_plus_2[0] = V[i0, i1 + 2]
		with hcl.elif_(i1 == 1):
			V_i_minus_1[0] = V[i0, i1 - 1]
			V_i_minus_2[0] = V[i0, i1] + my_abs(V[i0, i1] - V[i0, i1 - 1]) * my_sign(V[i0, i1])
			V_i_plus_1[0] = V[i0, i1 + 1]
			V_i_plus_2[0] = V[i0, i1 + 2]
		with hcl.elif_(i1 == V.shape[1] - 1):
			V_i_minus_1[0] = V[i0, i1 - 1]
			V_i_minus_2[0] = V[i0, i1 - 2]
			V_i_plus_1[0] = V[i0, i1] + my_abs(V[i0, i1] - V[i0, i1 - 1]) * my_sign(V[i0, i1])
			V_i_plus_2[0] = V[i0, i1] + 2 * my_abs(V[i0, i1] - V[i0, i1 - 1]) * my_sign(V[i0, i1])
		with hcl.elif_(i1 == V.shape[1] - 2):
			V_i_minus_1[0] = V[i0, i1 - 1]
			V_i_minus_2[0] = V[i0, i1 - 2]
			V_i_plus_1[0] = V[i0, i1 + 1]
			V_i_plus_2[0] =V[i0, i1] + my_abs(V[i0, i1 + 1] - V[i0, i1]) * my_sign(V[i0, i1])
		with hcl.else_():
			V_i_minus_1[0] = V[i0, i1 - 1]
			V_i_minus_2[0] = V[i0, i1 - 2]
			V_i_plus_1[0] = V[i0, i1 + 1]
			V_i_plus_2[0] = V[i0, i1 + 2]
	else:
		with hcl.if_(i1 == 0):
			V_i_minus_1[0] = V[i0, V.shape[1] - 1]
			V_i_minus_2[0] = V[i0, V.shape[1] - 2]
			V_i_plus_1[0] = V[i0, i1 + 1]
			V_i_plus_2[0] = V[i0, i1 + 2]
		with hcl.elif_(i1 == 1):
			V_i_minus_1[0] = V[i0, i1 - 1]
			V_i_minus_2[0] = V[i0, V.shape[1] - 1]
			V_i_plus_1[0] = V[i0, i1 + 1]
			V_i_plus_2[0] = V[i0, i1 + 2]
		with hcl.elif_(i1 == V.shape[1] - 1):
			V_i_minus_1[0] = V[i0, i1 - 1]
			V_i_minus_2[0] = V[i0, i1 - 2]
			V_i_plus_1[0] = V[i0, 0]
			V_i_plus_2[0] = V[i0, 1]
		with hcl.elif_(i1 == V.shape[1] - 2):
			V_i_minus_1[0] = V[i0, i1 - 1]
			V_i_minus_2[0] = V[i0, i1 - 2]
			V_i_plus_1[0] = V[i0, i1 + 1]
			V_i_plus_2[0] = V[i0, 0]
		with hcl.else_():
			V_i_minus_1[0] = V[i0, i1 - 1]
			V_i_minus_2[0] = V[i0, i1 - 2]
			V_i_plus_1[0] = V[i0, i1 + 1]
			V_i_plus_2[0] = V[i0, i1 + 2]
	D1_minus_2_plus_half = (V_i_minus_1[0] - V_i_minus_2[0]) / axis_step
	D1_minus_1_plus_half = (V[i0, i1] - V_i_minus_1[0]) / axis_step
	D1_0_plus_half = (V_i_plus_1[0] - V[i0, i1]) / axis_step
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

