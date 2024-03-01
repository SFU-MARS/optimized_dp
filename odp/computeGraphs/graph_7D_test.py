import heterocl as hcl
from odp.computeGraphs.CustomGraphFunctions import *
# from spatialDerivatives.second_orderENO7D_test import *
from odp.spatialDerivatives.first_orderENO7D_test import *

########################## 7D graph definition ######################## 
def graph_7D(my_object, g, compMethod, accuracy):
	V_f = hcl.placeholder(tuple(g.pts_each_dim), name="V_f", dtype=hcl.Float())
	V_init = hcl.placeholder(tuple(g.pts_each_dim), name="V_init", dtype=hcl.Float())
	l0 = hcl.placeholder(tuple(g.pts_each_dim), name="l0", dtype=hcl.Float())
	t = hcl.placeholder((2,), name="t", dtype=hcl.Float())
	
	x0 = hcl.placeholder((g.pts_each_dim[0],), name="x0", dtype=hcl.Float())
	x1 = hcl.placeholder((g.pts_each_dim[1],), name="x1", dtype=hcl.Float())
	x2 = hcl.placeholder((g.pts_each_dim[2],), name="x2", dtype=hcl.Float())
	x3 = hcl.placeholder((g.pts_each_dim[3],), name="x3", dtype=hcl.Float())
	x4 = hcl.placeholder((g.pts_each_dim[4],), name="x4", dtype=hcl.Float())
	x5 = hcl.placeholder((g.pts_each_dim[5],), name="x5", dtype=hcl.Float())
	x6 = hcl.placeholder((g.pts_each_dim[6],), name="x6", dtype=hcl.Float())

	def graph_create(V_new, V_init, x0, x1, x2, x3, x4, x5, x6, t, l0):
		# deriv_diff0 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff0")
		# deriv_diff1 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff1")
		# deriv_diff2 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff2")
		# deriv_diff3 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff3")
		# deriv_diff4 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff4")
		# deriv_diff5 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff5")
		# deriv_diff6 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff6")
		# max_deriv0 = hcl.scalar(my_object.speed_a, "max_deriv0")
		# max_deriv1 = hcl.scalar(my_object.speed_a, "max_deriv1")
		# max_deriv2 = hcl.scalar(my_object.speed_d, "max_deriv2")
		# max_deriv3 = hcl.scalar(my_object.speed_d, "max_deriv3")
		# max_deriv4 = hcl.scalar(my_object.speed_d, "max_deriv4")
		# max_deriv5 = hcl.scalar(my_object.speed_d, "max_deriv5")
		# max_deriv6 = hcl.scalar(my_object.speed_d, "max_deriv6")

		# min_deriv0 = hcl.scalar(1e9, "min_deriv0")
		# min_deriv1 = hcl.scalar(1e9, "min_deriv1")
		# min_deriv2 = hcl.scalar(1e9, "min_deriv2")
		# min_deriv3 = hcl.scalar(1e9, "min_deriv3")
		# min_deriv4 = hcl.scalar(1e9, "min_deriv4")
		# min_deriv5 = hcl.scalar(1e9, "min_deriv5")
		# min_deriv6 = hcl.scalar(1e9, "min_deriv6")
		#
		max_alpha0 = hcl.scalar(my_object.speed_a, "max_alpha0")
		max_alpha1 = hcl.scalar(my_object.speed_a, "max_alpha1")
		max_alpha2 = hcl.scalar(my_object.speed_d, "max_alpha2")
		max_alpha3 = hcl.scalar(my_object.speed_d, "max_alpha3")
		max_alpha4 = hcl.scalar(my_object.speed_d, "max_alpha4")
		max_alpha5 = hcl.scalar(my_object.speed_d, "max_alpha5")
		max_alpha6 = hcl.scalar(my_object.speed_d, "max_alpha6")

		def step_bound(): #Function to calculate time step
			stepBoundInv = hcl.scalar(0, "stepBoundInv")
			stepBound = hcl.scalar(0, "stepBound")
			stepBoundInv[0] = max_alpha0[0] / g.dx[0]+ max_alpha1[0] / g.dx[1]+ max_alpha2[0] / g.dx[2]+ max_alpha3[0] / g.dx[3]+ max_alpha4[0] / g.dx[4]+ max_alpha5[0] / g.dx[5]+ max_alpha6[0] / g.dx[6]

			stepBound[0] = 0.8 / stepBoundInv[0]
			with hcl.if_(stepBound > t[1] - t[0]):
				stepBound[0] = t[1] - t[0]

			# Update the lower time range
			t[0] = t[0] + stepBound[0]
			return stepBound[0]

		def minVWithVInit(i0, i1, i2, i3, i4, i5, i6):
			with hcl.if_(V_new[i0, i1, i2, i3, i4, i5, i6] > V_init[i0, i1, i2, i3, i4, i5, i6]):
				V_new[i0, i1, i2, i3, i4, i5, i6] = V_init[i0, i1, i2, i3, i4, i5, i6]
		def maxVWithVInit(i0, i1, i2, i3, i4, i5, i6):
			with hcl.if_(V_new[i0, i1, i2, i3, i4, i5, i6] < V_init[i0, i1, i2, i3, i4, i5, i6]):
				V_new[i0, i1, i2, i3, i4, i5, i6] = V_init[i0, i1, i2, i3, i4, i5, i6]
		def maxVWithV0(i0, i1, i2, i3, i4, i5, i6):
			with hcl.if_(V_new[i0, i1, i2, i3, i4, i5, i6] < l0[i0, i1, i2, i3, i4, i5, i6]):
				V_new[i0, i1, i2, i3, i4, i5, i6] = l0[i0, i1, i2, i3, i4, i5, i6]
		def minVWithV0(i0, i1, i2, i3, i4, i5, i6):
			with hcl.if_(V_new[i0, i1, i2, i3, i4, i5, i6] > l0[i0, i1, i2, i3, i4, i5, i6]):
				V_new[i0, i1, i2, i3, i4, i5, i6] = l0[i0, i1, i2, i3, i4, i5, i6]

		with hcl.Stage("Hamiltonian"):
			with hcl.for_(0, V_init.shape[0], name="i0") as i0:
				with hcl.for_(0, V_init.shape[1], name="i1") as i1:
					with hcl.for_(0, V_init.shape[2], name="i2") as i2:
						with hcl.for_(0, V_init.shape[3], name="i3") as i3:
							with hcl.for_(0, V_init.shape[4], name="i4") as i4:
								with hcl.for_(0, V_init.shape[5], name="i5") as i5:
									with hcl.for_(0, V_init.shape[6], name="i6") as i6:
										dV_dx_L0 = hcl.scalar(0, "dV_dx_L0")
										dV_dx_R0 = hcl.scalar(0, "dV_dx_R0")
										dV_dx0 = hcl.scalar(0, "dV_dx0")
										dV_dx_L1 = hcl.scalar(0, "dV_dx_L1")
										dV_dx_R1 = hcl.scalar(0, "dV_dx_R1")
										dV_dx1 = hcl.scalar(0, "dV_dx1")
										dV_dx_L2 = hcl.scalar(0, "dV_dx_L2")
										dV_dx_R2 = hcl.scalar(0, "dV_dx_R2")
										dV_dx2 = hcl.scalar(0, "dV_dx2")
										dV_dx_L3 = hcl.scalar(0, "dV_dx_L3")
										dV_dx_R3 = hcl.scalar(0, "dV_dx_R3")
										dV_dx3 = hcl.scalar(0, "dV_dx3")
										dV_dx_L4 = hcl.scalar(0, "dV_dx_L4")
										dV_dx_R4 = hcl.scalar(0, "dV_dx_R4")
										dV_dx4 = hcl.scalar(0, "dV_dx4")
										dV_dx_L5 = hcl.scalar(0, "dV_dx_L5")
										dV_dx_R5 = hcl.scalar(0, "dV_dx_R5")
										dV_dx5 = hcl.scalar(0, "dV_dx5")
										dV_dx_L6 = hcl.scalar(0, "dV_dx_L6")
										dV_dx_R6 = hcl.scalar(0, "dV_dx_R6")
										dV_dx6 = hcl.scalar(0, "dV_dx6")
										if accuracy == "low":
											dV_dx_L0[0], dV_dx_R0[0] = spa_derivX0_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L1[0], dV_dx_R1[0] = spa_derivX1_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L2[0], dV_dx_R2[0] = spa_derivX2_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L3[0], dV_dx_R3[0] = spa_derivX3_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L4[0], dV_dx_R4[0] = spa_derivX4_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L5[0], dV_dx_R5[0] = spa_derivX5_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L6[0], dV_dx_R6[0] = spa_derivX6_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
										if accuracy == "medium":
											dV_dx_L0[0], dV_dx_R0[0] = secondOrderX0_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L1[0], dV_dx_R1[0] = secondOrderX1_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L2[0], dV_dx_R2[0] = secondOrderX2_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L3[0], dV_dx_R3[0] = secondOrderX3_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L4[0], dV_dx_R4[0] = secondOrderX4_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L5[0], dV_dx_R5[0] = secondOrderX5_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)
											dV_dx_L6[0], dV_dx_R6[0] = secondOrderX6_7d(i0, i1, i2, i3, i4, i5, i6, V_init, g)

										# deriv_diff0[i0, i1, i2, i3, i4, i5, i6] = dV_dx_R0[0] - dV_dx_L0[0]
										# deriv_diff1[i0, i1, i2, i3, i4, i5, i6] = dV_dx_R1[0] - dV_dx_L1[0]
										# deriv_diff2[i0, i1, i2, i3, i4, i5, i6] = dV_dx_R2[0] - dV_dx_L2[0]
										# deriv_diff3[i0, i1, i2, i3, i4, i5, i6] = dV_dx_R3[0] - dV_dx_L3[0]
										# deriv_diff4[i0, i1, i2, i3, i4, i5, i6] = dV_dx_R4[0] - dV_dx_L4[0]
										# deriv_diff5[i0, i1, i2, i3, i4, i5, i6] = dV_dx_R5[0] - dV_dx_L5[0]
										# deriv_diff6[i0, i1, i2, i3, i4, i5, i6] = dV_dx_R6[0] - dV_dx_L6[0]
										deriv_diff0 = hcl.scalar(0, "deriv_diff0")
										deriv_diff1 = hcl.scalar(0, "deriv_diff1")
										deriv_diff2 = hcl.scalar(0, "deriv_diff2")
										deriv_diff3 = hcl.scalar(0, "deriv_diff3")
										deriv_diff4 = hcl.scalar(0, "deriv_diff4")
										deriv_diff5 = hcl.scalar(0, "deriv_diff5")
										deriv_diff6 = hcl.scalar(0, "deriv_diff6")

										deriv_diff0[0] = dV_dx_R0[0] - dV_dx_L0[0]
										deriv_diff1[0] = dV_dx_R1[0] - dV_dx_L1[0]
										deriv_diff2[0] = dV_dx_R2[0] - dV_dx_L2[0]
										deriv_diff3[0] = dV_dx_R3[0] - dV_dx_L3[0]
										deriv_diff4[0] = dV_dx_R4[0] - dV_dx_L4[0]
										deriv_diff5[0] = dV_dx_R5[0] - dV_dx_L5[0]
										deriv_diff6[0] = dV_dx_R6[0] - dV_dx_L6[0]

										dV_dx0[0] = (dV_dx_L0 + dV_dx_R0) / 2
										dV_dx1[0] = (dV_dx_L1 + dV_dx_R1) / 2
										dV_dx2[0] = (dV_dx_L2 + dV_dx_R2) / 2
										dV_dx3[0] = (dV_dx_L3 + dV_dx_R3) / 2
										dV_dx4[0] = (dV_dx_L4 + dV_dx_R4) / 2
										dV_dx5[0] = (dV_dx_L5 + dV_dx_R5) / 2
										dV_dx6[0] = (dV_dx_L6 + dV_dx_R6) / 2
										uOpt = my_object.opt_ctrl(t, (x0[i0], x1[i1], x2[i2], x3[i3], x4[i4], x5[i5], x6[i6]),
 (dV_dx0[0], dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0], dV_dx5[0], dV_dx6[0]))
										dOpt = my_object.opt_dstb(t, (x0[i0], x1[i1], x2[i2], x3[i3], x4[i4], x5[i5], x6[i6]),
 (dV_dx0[0], dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0], dV_dx5[0], dV_dx6[0]))
										dx_dt0, dx_dt1, dx_dt2, dx_dt3, dx_dt4, dx_dt5, dx_dt6 = my_object.dynamics(t, (x0[i0], x1[i1], x2[i2], x3[i3], x4[i4], x5[i5], x6[i6]), uOpt, dOpt)
										V_new[i0, i1, i2, i3, i4, i5, i6] = -(dx_dt0 * dV_dx0[0] + dx_dt1 * dV_dx1[0] + dx_dt2 * dV_dx2[0] + dx_dt3 * dV_dx3[0] + dx_dt4 * dV_dx4[0] + dx_dt5 * dV_dx5[0] + dx_dt6 * dV_dx6[0])

										# Directly add the dissipation here
										alpha0 = hcl.scalar(my_object.speed_a, "alpha1")
										alpha1 = hcl.scalar(my_object.speed_a, "alpha2")
										alpha2 = hcl.scalar(my_object.speed_d, "alpha3")
										alpha3 = hcl.scalar(my_object.speed_d, "alpha4")
										alpha4 = hcl.scalar(my_object.speed_d, "alpha5")
										alpha5 = hcl.scalar(my_object.speed_d, "alpha6")
										alpha6 = hcl.scalar(my_object.speed_d, "alpha7")

										diss = hcl.scalar(0, "diss")
										diss[0] = 0.5 * (deriv_diff0[0] * alpha0[0] + deriv_diff1[0] * alpha1[0] + deriv_diff2[0] * alpha2[0] \
														 + deriv_diff3[0] * alpha3[0] + deriv_diff4[0] * alpha4[0] \
														 + deriv_diff5[0] * alpha5[0] + deriv_diff6[0] * alpha6[0])

										# Finally
										V_new[i0, i1, i2, i3, i4, i5, i6] = -(V_new[i0, i1, i2, i3, i4, i5, i6] - diss[0])

										# with hcl.if_(dV_dx_L0[0] < min_deriv0[0]):
										# 	min_deriv0[0] = dV_dx_L0[0]
										# with hcl.if_(dV_dx_L0[0] > max_deriv0[0]):
										# 	max_deriv0[0] = dV_dx_L0[0]
										# with hcl.if_(dV_dx_R0[0] < min_deriv0[0]):
										# 	min_deriv0[0] = dV_dx_R0[0]
										# with hcl.if_(dV_dx_R0[0] > max_deriv0[0]):
										# 	max_deriv0[0] = dV_dx_R0[0]
										# with hcl.if_(dV_dx_L1[0] < min_deriv1[0]):
										# 	min_deriv1[0] = dV_dx_L1[0]
										# with hcl.if_(dV_dx_L1[0] > max_deriv1[0]):
										# 	max_deriv1[0] = dV_dx_L1[0]
										# with hcl.if_(dV_dx_R1[0] < min_deriv1[0]):
										# 	min_deriv1[0] = dV_dx_R1[0]
										# with hcl.if_(dV_dx_R1[0] > max_deriv1[0]):
										# 	max_deriv1[0] = dV_dx_R1[0]
										# with hcl.if_(dV_dx_L2[0] < min_deriv2[0]):
										# 	min_deriv2[0] = dV_dx_L2[0]
										# with hcl.if_(dV_dx_L2[0] > max_deriv2[0]):
										# 	max_deriv2[0] = dV_dx_L2[0]
										# with hcl.if_(dV_dx_R2[0] < min_deriv2[0]):
										# 	min_deriv2[0] = dV_dx_R2[0]
										# with hcl.if_(dV_dx_R2[0] > max_deriv2[0]):
										# 	max_deriv2[0] = dV_dx_R2[0]
										# with hcl.if_(dV_dx_L3[0] < min_deriv3[0]):
										# 	min_deriv3[0] = dV_dx_L3[0]
										# with hcl.if_(dV_dx_L3[0] > max_deriv3[0]):
										# 	max_deriv3[0] = dV_dx_L3[0]
										# with hcl.if_(dV_dx_R3[0] < min_deriv3[0]):
										# 	min_deriv3[0] = dV_dx_R3[0]
										# with hcl.if_(dV_dx_R3[0] > max_deriv3[0]):
										# 	max_deriv3[0] = dV_dx_R3[0]
										# with hcl.if_(dV_dx_L4[0] < min_deriv4[0]):
										# 	min_deriv4[0] = dV_dx_L4[0]
										# with hcl.if_(dV_dx_L4[0] > max_deriv4[0]):
										# 	max_deriv4[0] = dV_dx_L4[0]
										# with hcl.if_(dV_dx_R4[0] < min_deriv4[0]):
										# 	min_deriv4[0] = dV_dx_R4[0]
										# with hcl.if_(dV_dx_R4[0] > max_deriv4[0]):
										# 	max_deriv4[0] = dV_dx_R4[0]
										# with hcl.if_(dV_dx_L5[0] < min_deriv5[0]):
										# 	min_deriv5[0] = dV_dx_L5[0]
										# with hcl.if_(dV_dx_L5[0] > max_deriv5[0]):
										# 	max_deriv5[0] = dV_dx_L5[0]
										# with hcl.if_(dV_dx_R5[0] < min_deriv5[0]):
										# 	min_deriv5[0] = dV_dx_R5[0]
										# with hcl.if_(dV_dx_R5[0] > max_deriv5[0]):
										# 	max_deriv5[0] = dV_dx_R5[0]
										# with hcl.if_(dV_dx_L6[0] < min_deriv6[0]):
										# 	min_deriv6[0] = dV_dx_L6[0]
										# with hcl.if_(dV_dx_L6[0] > max_deriv6[0]):
										# 	max_deriv6[0] = dV_dx_L6[0]
										# with hcl.if_(dV_dx_R6[0] < min_deriv6[0]):
										# 	min_deriv6[0] = dV_dx_R6[0]
										# with hcl.if_(dV_dx_R6[0] > max_deriv6[0]):
										# 	max_deriv6[0] = dV_dx_R6[0]

# 		with hcl.Stage("Dissipation"):
# 			uOptL0 = hcl.scalar(0, "uOptL0")
# 			uOptL1 = hcl.scalar(0, "uOptL1")
# 			uOptL2 = hcl.scalar(0, "uOptL2")
# 			uOptL3 = hcl.scalar(0, "uOptL3")
# 			uOptL4 = hcl.scalar(0, "uOptL4")
# 			uOptL5 = hcl.scalar(0, "uOptL5")
# 			uOptL6 = hcl.scalar(0, "uOptL6")
# 			uOptL7 = hcl.scalar(0, "uOptL7")
# 			uOptU0 = hcl.scalar(0, "uOptU0")
# 			uOptU1 = hcl.scalar(0, "uOptU1")
# 			uOptU2 = hcl.scalar(0, "uOptU2")
# 			uOptU3 = hcl.scalar(0, "uOptU3")
# 			uOptU4 = hcl.scalar(0, "uOptU4")
# 			uOptU5 = hcl.scalar(0, "uOptU5")
# 			uOptU6 = hcl.scalar(0, "uOptU6")
# 			uOptU7 = hcl.scalar(0, "uOptU7")
# 			dOptL0 = hcl.scalar(0, "dOptL0")
# 			dOptL1 = hcl.scalar(0, "dOptL1")
# 			dOptL2 = hcl.scalar(0, "dOptL2")
# 			dOptL3 = hcl.scalar(0, "dOptL3")
# 			dOptL4 = hcl.scalar(0, "dOptL4")
# 			dOptL5 = hcl.scalar(0, "dOptL5")
# 			dOptL6 = hcl.scalar(0, "dOptL6")
# 			dOptL7 = hcl.scalar(0, "dOptL7")
# 			dOptU0 = hcl.scalar(0, "dOptU0")
# 			dOptU1 = hcl.scalar(0, "dOptU1")
# 			dOptU2 = hcl.scalar(0, "dOptU2")
# 			dOptU3 = hcl.scalar(0, "dOptU3")
# 			dOptU4 = hcl.scalar(0, "dOptU4")
# 			dOptU5 = hcl.scalar(0, "dOptU5")
# 			dOptU6 = hcl.scalar(0, "dOptU6")
# 			dOptU7 = hcl.scalar(0, "dOptU7")
# 			alpha0 = hcl.scalar(0, "alpha0")
# 			alpha1 = hcl.scalar(0, "alpha1")
# 			alpha2 = hcl.scalar(0, "alpha2")
# 			alpha3 = hcl.scalar(0, "alpha3")
# 			alpha4 = hcl.scalar(0, "alpha4")
# 			alpha5 = hcl.scalar(0, "alpha5")
# 			alpha6 = hcl.scalar(0, "alpha6")
# 			with hcl.for_(0, V_init.shape[0], name="i0") as i0:
# 				with hcl.for_(0, V_init.shape[1], name="i1") as i1:
# 					with hcl.for_(0, V_init.shape[2], name="i2") as i2:
# 						with hcl.for_(0, V_init.shape[3], name="i3") as i3:
# 							with hcl.for_(0, V_init.shape[4], name="i4") as i4:
# 								with hcl.for_(0, V_init.shape[5], name="i5") as i5:
# 									with hcl.for_(0, V_init.shape[6], name="i6") as i6:
# 										dx_UU0 = hcl.scalar(0, "dx_UU0")
# 										dx_UU1 = hcl.scalar(0, "dx_UU1")
# 										dx_UU2 = hcl.scalar(0, "dx_UU2")
# 										dx_UU3 = hcl.scalar(0, "dx_UU3")
# 										dx_UU4 = hcl.scalar(0, "dx_UU4")
# 										dx_UU5 = hcl.scalar(0, "dx_UU5")
# 										dx_UU6 = hcl.scalar(0, "dx_UU6")
# 										dx_UL0 = hcl.scalar(0, "dx_UL0")
# 										dx_UL1 = hcl.scalar(0, "dx_UL1")
# 										dx_UL2 = hcl.scalar(0, "dx_UL2")
# 										dx_UL3 = hcl.scalar(0, "dx_UL3")
# 										dx_UL4 = hcl.scalar(0, "dx_UL4")
# 										dx_UL5 = hcl.scalar(0, "dx_UL5")
# 										dx_UL6 = hcl.scalar(0, "dx_UL6")
# 										dx_LU0 = hcl.scalar(0, "dx_LU0")
# 										dx_LU1 = hcl.scalar(0, "dx_LU1")
# 										dx_LU2 = hcl.scalar(0, "dx_LU2")
# 										dx_LU3 = hcl.scalar(0, "dx_LU3")
# 										dx_LU4 = hcl.scalar(0, "dx_LU4")
# 										dx_LU5 = hcl.scalar(0, "dx_LU5")
# 										dx_LU6 = hcl.scalar(0, "dx_LU6")
# 										dx_LL0 = hcl.scalar(0, "dx_LL0")
# 										dx_LL1 = hcl.scalar(0, "dx_LL1")
# 										dx_LL2 = hcl.scalar(0, "dx_LL2")
# 										dx_LL3 = hcl.scalar(0, "dx_LL3")
# 										dx_LL4 = hcl.scalar(0, "dx_LL4")
# 										dx_LL5 = hcl.scalar(0, "dx_LL5")
# 										dx_LL6 = hcl.scalar(0, "dx_LL6")
#
# 										dOptL0[0], dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0], dOptL5[0], dOptL6[0], dOptL7[0] = my_object.opt_dstb(t, (x0[i0], x1[i1], x2[i2], x3[i3], x4[i4], x5[i5], x6[i6]),
# (min_deriv0[0], min_deriv1[0], min_deriv2[0], min_deriv3[0], min_deriv4[0], min_deriv5[0], min_deriv6[0]) )
# 										dOptU0[0], dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0], dOptU5[0], dOptU6[0], dOptU7[0] = my_object.opt_dstb(t, (x0[i0], x1[i1], x2[i2], x3[i3], x4[i4], x5[i5], x6[i6]),
# (max_deriv0[0], max_deriv1[0], max_deriv2[0], max_deriv3[0], max_deriv4[0], max_deriv5[0], max_deriv6[0]) )
# 										uOptL0[0], uOptL1[0], uOptL2[0], uOptL3[0], uOptL4[0], uOptL5[0], uOptL6[0], uOptL7[0] = my_object.opt_ctrl(t, (x0[i0], x1[i1], x2[i2], x3[i3], x4[i4], x5[i5], x6[i6]),
# (min_deriv0[0], min_deriv1[0], min_deriv2[0], min_deriv3[0], min_deriv4[0], min_deriv5[0], min_deriv6[0]) )
# 										uOptU0[0], uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0], uOptU5[0], uOptU6[0], uOptU7[0] = my_object.opt_ctrl(t, (x0[i0], x1[i1], x2[i2], x3[i3], x4[i4], x5[i5], x6[i6]),
# (max_deriv0[0], max_deriv1[0], max_deriv2[0], max_deriv3[0], max_deriv4[0], max_deriv5[0], max_deriv6[0]) )
# 										dx_LL0[0], dx_LL1[0], dx_LL2[0], dx_LL3[0], dx_LL4[0], dx_LL5[0], dx_LL6[0] = my_object.dynamics(t, (x0[i0], x1[i1], x2[i2], x3[i3], x4[i4], x5[i5], x6[i6]),
# (uOptL0[0], uOptL1[0], uOptL2[0], uOptL3[0], uOptL4[0], uOptL5[0], uOptL6[0], uOptL7[0]),
# (dOptL0[0], dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0], dOptL5[0], dOptL6[0], dOptL7[0]))
# 										dx_LL0[0] = my_abs(dx_LL0[0])
# 										dx_LL1[0] = my_abs(dx_LL1[0])
# 										dx_LL2[0] = my_abs(dx_LL2[0])
# 										dx_LL3[0] = my_abs(dx_LL3[0])
# 										dx_LL4[0] = my_abs(dx_LL4[0])
# 										dx_LL5[0] = my_abs(dx_LL5[0])
# 										dx_LL6[0] = my_abs(dx_LL6[0])
# 										dx_UL0[0], dx_UL1[0], dx_UL2[0], dx_UL3[0], dx_UL4[0], dx_UL5[0], dx_UL6[0] = my_object.dynamics(t, (x0[i0], x1[i1], x2[i2], x3[i3], x4[i4], x5[i5], x6[i6]),
# (uOptU0[0], uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0], uOptU5[0], uOptU6[0], uOptU7[0]),
# (dOptL0[0], dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0], dOptL5[0], dOptL6[0], dOptL7[0]))
# 										dx_UL0[0] = my_abs(dx_UL0[0])
# 										dx_UL1[0] = my_abs(dx_UL1[0])
# 										dx_UL2[0] = my_abs(dx_UL2[0])
# 										dx_UL3[0] = my_abs(dx_UL3[0])
# 										dx_UL4[0] = my_abs(dx_UL4[0])
# 										dx_UL5[0] = my_abs(dx_UL5[0])
# 										dx_UL6[0] = my_abs(dx_UL6[0])
# 										alpha0[0] = my_max(dx_UL0[0], dx_LL0[0])
# 										alpha1[0] = my_max(dx_UL1[0], dx_LL1[0])
# 										alpha2[0] = my_max(dx_UL2[0], dx_LL2[0])
# 										alpha3[0] = my_max(dx_UL3[0], dx_LL3[0])
# 										alpha4[0] = my_max(dx_UL4[0], dx_LL4[0])
# 										alpha5[0] = my_max(dx_UL5[0], dx_LL5[0])
# 										alpha6[0] = my_max(dx_UL6[0], dx_LL6[0])
# 										dx_LU0[0], dx_LU1[0], dx_LU2[0], dx_LU3[0], dx_LU4[0], dx_LU5[0], dx_LU6[0] = my_object.dynamics(t, (x0[i0], x1[i1], x2[i2], x3[i3], x4[i4], x5[i5], x6[i6]),
# (uOptL0[0], uOptL1[0], uOptL2[0], uOptL3[0], uOptL4[0], uOptL5[0], uOptL6[0], uOptL7[0]),
# (dOptU0[0], dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0], dOptU5[0], dOptU6[0], dOptU7[0]))
# 										dx_LU0[0] = my_abs(dx_LU0[0])
# 										dx_LU1[0] = my_abs(dx_LU1[0])
# 										dx_LU2[0] = my_abs(dx_LU2[0])
# 										dx_LU3[0] = my_abs(dx_LU3[0])
# 										dx_LU4[0] = my_abs(dx_LU4[0])
# 										dx_LU5[0] = my_abs(dx_LU5[0])
# 										dx_LU6[0] = my_abs(dx_LU6[0])
# 										alpha0[0] = my_max(alpha0[0], dx_LU0[0])
# 										alpha1[0] = my_max(alpha1[0], dx_LU1[0])
# 										alpha2[0] = my_max(alpha2[0], dx_LU2[0])
# 										alpha3[0] = my_max(alpha3[0], dx_LU3[0])
# 										alpha4[0] = my_max(alpha4[0], dx_LU4[0])
# 										alpha5[0] = my_max(alpha5[0], dx_LU5[0])
# 										alpha6[0] = my_max(alpha6[0], dx_LU6[0])
# 										dx_UU0[0], dx_UU1[0], dx_UU2[0], dx_UU3[0], dx_UU4[0], dx_UU5[0], dx_UU6[0] = my_object.dynamics(t, (x0[i0], x1[i1], x2[i2], x3[i3], x4[i4], x5[i5], x6[i6]),
# (uOptU0[0], uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0], uOptU5[0], uOptU6[0], uOptU7[0]),
# (dOptU0[0], dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0], dOptU5[0], dOptU6[0], dOptU7[0]))
# 										dx_UU0[0] = my_abs(dx_UU0[0])
# 										dx_UU1[0] = my_abs(dx_UU1[0])
# 										dx_UU2[0] = my_abs(dx_UU2[0])
# 										dx_UU3[0] = my_abs(dx_UU3[0])
# 										dx_UU4[0] = my_abs(dx_UU4[0])
# 										dx_UU5[0] = my_abs(dx_UU5[0])
# 										dx_UU6[0] = my_abs(dx_UU6[0])
# 										alpha0[0] = my_max(alpha0[0], dx_UU0[0])
# 										alpha1[0] = my_max(alpha1[0], dx_UU1[0])
# 										alpha2[0] = my_max(alpha2[0], dx_UU2[0])
# 										alpha3[0] = my_max(alpha3[0], dx_UU3[0])
# 										alpha4[0] = my_max(alpha4[0], dx_UU4[0])
# 										alpha5[0] = my_max(alpha5[0], dx_UU5[0])
# 										alpha6[0] = my_max(alpha6[0], dx_UU6[0])
# 										diss = hcl.scalar(0, "diss")
# 										diss[0] = 0.5 * (deriv_diff0[i0, i1, i2, i3, i4, i5, i6] * alpha0[0] + deriv_diff1[i0, i1, i2, i3, i4, i5, i6] * alpha1[0] + deriv_diff2[i0, i1, i2, i3, i4, i5, i6] * alpha2[0] + deriv_diff3[i0, i1, i2, i3, i4, i5, i6] * alpha3[0] + deriv_diff4[i0, i1, i2, i3, i4, i5, i6] * alpha4[0] + deriv_diff5[i0, i1, i2, i3, i4, i5, i6] * alpha5[0] + deriv_diff6[i0, i1, i2, i3, i4, i5, i6] * alpha6[0])
# 										V_new[i0, i1, i2, i3, i4, i5, i6] = -(V_new[i0, i1, i2, i3, i4, i5, i6] - diss[0])
# 										with hcl.if_(alpha0[0] > max_alpha0[0]):
# 											max_alpha0[0] = alpha0[0]
# 										with hcl.if_(alpha1[0] > max_alpha1[0]):
# 											max_alpha1[0] = alpha1[0]
# 										with hcl.if_(alpha2[0] > max_alpha2[0]):
# 											max_alpha2[0] = alpha2[0]
# 										with hcl.if_(alpha3[0] > max_alpha3[0]):
# 											max_alpha3[0] = alpha3[0]
# 										with hcl.if_(alpha4[0] > max_alpha4[0]):
# 											max_alpha4[0] = alpha4[0]
# 										with hcl.if_(alpha5[0] > max_alpha5[0]):
# 											max_alpha5[0] = alpha5[0]
# 										with hcl.if_(alpha6[0] > max_alpha6[0]):
# 											max_alpha6[0] = alpha6[0]
		delta_t = hcl.compute((1,), lambda x: step_bound(), name="delta_t")
		result = hcl.update(V_new, lambda i0, i1, i2, i3, i4, i5, i6: V_init[i0, i1, i2, i3, i4, i5, i6] + V_new[i0, i1, i2, i3, i4, i5, i6] * delta_t[0])
		if compMethod == 'maxVWithV0' or compMethod == 'maxVWithVTarget':
			result = hcl.update(V_new, lambda i0, i1, i2, i3, i4, i5, i6:  maxVWithV0(i0, i1, i2, i3, i4, i5, i6))
		if compMethod == 'minVWithV0' or compMethod == 'minVWithVTarget':
			result = hcl.update(V_new, lambda i0, i1, i2, i3, i4, i5, i6:  minVWithV0(i0, i1, i2, i3, i4, i5, i6))
		if compMethod == 'maxVWithVInit' :
			result = hcl.update(V_new, lambda i0, i1, i2, i3, i4, i5, i6:  maxVWithVInit(i0, i1, i2, i3, i4, i5, i6))
		if compMethod == 'minVWithVInit' :
			result = hcl.update(V_new, lambda i0, i1, i2, i3, i4, i5, i6:  minVWithVInit(i0, i1, i2, i3, i4, i5, i6))
		hcl.update(V_init, lambda i0, i1, i2, i3, i4, i5, i6:  V_new[i0, i1, i2, i3, i4, i5, i6])
		return result
	s = hcl.create_schedule([V_f, V_init, x0, x1, x2, x3, x4, x5, x6, t, l0], graph_create)
	#Optimizing

	s_H = graph_create.Hamiltonian
	# s_D = graph_create.Dissipation

	s[s_H].parallel(s_H.i0)
	# s[s_D].parallel(s_D.i0)

	return (hcl.build(s))