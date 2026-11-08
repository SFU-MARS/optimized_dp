import heterocl as hcl
import numpy as np
from odp.computeGraphs.CustomGraphFunctions import *
from odp.spatialDerivatives.firstOrderENO.first_orderENO4D import *
from odp.spatialDerivatives.secondOrderENO.second_orderENO4D import *

########################## 4D Graph definition #################################
def graph_4D(my_object, g, compMethod, accuracy, generate_SpatDeriv=False, deriv_dim=1):
    V_f = hcl.placeholder(tuple(g.pts_each_dim), name="V_f", dtype=hcl.Float())
    V_init = hcl.placeholder(tuple(g.pts_each_dim), name="V_init", dtype=hcl.Float())
    l0 = hcl.placeholder(tuple(g.pts_each_dim), name="l0", dtype=hcl.Float())
    t = hcl.placeholder((1,), name="t", dtype=hcl.Float())
    delta_t = hcl.placeholder((1,), name="delta_t", dtype=hcl.Float())
    #probe = hcl.placeholder(tuple(g.pts_each_dim), name="probe", dtype=hcl.Float())

    # Positions vector
    x1 = hcl.placeholder((g.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((g.pts_each_dim[1],), name="x2", dtype=hcl.Float())
    x3 = hcl.placeholder((g.pts_each_dim[2],), name="x3", dtype=hcl.Float())
    x4 = hcl.placeholder((g.pts_each_dim[3],), name="x4", dtype=hcl.Float())

    def graph_create(V_new, V_init, x1, x2, x3, x4, delta_t, t, l0):
        # Specify intermediate tensors
        deriv_diff1 = hcl.compute(V_init.shape, lambda *x:0, "deriv_diff1")
        deriv_diff2 = hcl.compute(V_init.shape, lambda *x:0, "deriv_diff2")
        deriv_diff3 = hcl.compute(V_init.shape, lambda *x:0, "deriv_diff3")
        deriv_diff4 = hcl.compute(V_init.shape, lambda *x:0, "deriv_diff4")

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

        # Per-row (outermost index) running max/min, so threads of the parallel loops never write
        # the same location; reduced serially into the scalars above
        N0 = V_init.shape[0]
        max_deriv1_rows = hcl.compute((N0,), lambda x: -1e9, "max_deriv1_rows")
        max_deriv2_rows = hcl.compute((N0,), lambda x: -1e9, "max_deriv2_rows")
        max_deriv3_rows = hcl.compute((N0,), lambda x: -1e9, "max_deriv3_rows")
        max_deriv4_rows = hcl.compute((N0,), lambda x: -1e9, "max_deriv4_rows")
        max_alpha1_rows = hcl.compute((N0,), lambda x: -1e9, "max_alpha1_rows")
        max_alpha2_rows = hcl.compute((N0,), lambda x: -1e9, "max_alpha2_rows")
        max_alpha3_rows = hcl.compute((N0,), lambda x: -1e9, "max_alpha3_rows")
        max_alpha4_rows = hcl.compute((N0,), lambda x: -1e9, "max_alpha4_rows")
        min_deriv1_rows = hcl.compute((N0,), lambda x: 1e9, "min_deriv1_rows")
        min_deriv2_rows = hcl.compute((N0,), lambda x: 1e9, "min_deriv2_rows")
        min_deriv3_rows = hcl.compute((N0,), lambda x: 1e9, "min_deriv3_rows")
        min_deriv4_rows = hcl.compute((N0,), lambda x: 1e9, "min_deriv4_rows")

        def step_bound():  # Function to calculate time step
            stepBoundInv = hcl.scalar(0, "stepBoundInv")
            stepBound = hcl.scalar(0, "stepBound")
            stepBoundInv[0] = max_alpha1[0] / g.dx[0] + max_alpha2[0] / g.dx[1] + max_alpha3[0] / g.dx[2] + max_alpha4[0] / \
                              g.dx[3]
            stepBound[0] = 0.8 / stepBoundInv[0]
            return stepBound[0]

        # Calculate Hamiltonian for every grid point in V_init
        with hcl.Stage("Hamiltonian"):
            with hcl.for_(0, V_init.shape[0], name="i") as i:
                with hcl.for_(0, V_init.shape[1], name="j") as j:
                    with hcl.for_(0, V_init.shape[2], name="k") as k:
                        with hcl.for_(0, V_init.shape[3], name="l") as l:
                            # Variables to calculate dV_dx
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
                            # dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
                            if accuracy == "low":
                                dV_dx1_L[0], dV_dx1_R[0] = spa_derivX1_4d(i, j, k, l, V_init, g)
                                dV_dx2_L[0], dV_dx2_R[0] = spa_derivX2_4d(i, j, k, l, V_init, g)
                                dV_dx3_L[0], dV_dx3_R[0] = spa_derivX3_4d(i, j, k, l, V_init, g)
                                dV_dx4_L[0], dV_dx4_R[0] = spa_derivX4_4d(i, j, k, l, V_init, g)
                            if accuracy == "medium":
                                dV_dx1_L[0], dV_dx1_R[0] = secondOrder_ENO4D_X0(i, j, k, l, V_init, g)
                                dV_dx2_L[0], dV_dx2_R[0] = secondOrder_ENO4D_X1(i, j, k, l, V_init, g)
                                dV_dx3_L[0], dV_dx3_R[0] = secondOrder_ENO4D_X2(i, j, k, l, V_init, g)
                                dV_dx4_L[0], dV_dx4_R[0] = secondOrder_ENO4D_X3(i, j, k, l, V_init, g)

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

                            #probe[i,j,k,l] = dV_dx2[0]
                            # Find optimal control
                            uOpt = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k], x4[l]),
                                                      (dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0]))

                            # Find optimal disturbance
                            dOpt = my_object.opt_dstb(t, (x1[i], x2[j], x3[k], x4[l]),
                                                      (dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0]))

                            # Find rates of changes based on dynamics equation
                            dx1_dt, dx2_dt, dx3_dt, dx4_dt = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]), uOpt, dOpt)

                            # Calculate Hamiltonian terms:
                            V_new[i, j, k, l] = -(
                                        dx1_dt * dV_dx1[0] + dx2_dt * dV_dx2[0] + dx3_dt * dV_dx3[0] + dx4_dt * dV_dx4[0])
                            # Debugging
                            # V_new[i, j, k, l] = dV_dx2[0]
                            #probe[i, j, k, l] = V_init[i, j, k, l]

                            # Get derivMin
                            update_row_min(min_deriv1_rows, i, dV_dx1_L[0])
                            update_row_min(min_deriv1_rows, i, dV_dx1_R[0])

                            update_row_min(min_deriv2_rows, i, dV_dx2_L[0])
                            update_row_min(min_deriv2_rows, i, dV_dx2_R[0])

                            update_row_min(min_deriv3_rows, i, dV_dx3_L[0])
                            update_row_min(min_deriv3_rows, i, dV_dx3_R[0])

                            update_row_min(min_deriv4_rows, i, dV_dx4_L[0])
                            update_row_min(min_deriv4_rows, i, dV_dx4_R[0])

                            # Get derivMax
                            update_row_max(max_deriv1_rows, i, dV_dx1_L[0])
                            update_row_max(max_deriv1_rows, i, dV_dx1_R[0])

                            update_row_max(max_deriv2_rows, i, dV_dx2_L[0])
                            update_row_max(max_deriv2_rows, i, dV_dx2_R[0])

                            update_row_max(max_deriv3_rows, i, dV_dx3_L[0])
                            update_row_max(max_deriv3_rows, i, dV_dx3_R[0])

                            update_row_max(max_deriv4_rows, i, dV_dx4_L[0])
                            update_row_max(max_deriv4_rows, i, dV_dx4_R[0])

        # Calculate dissipation amount
        with hcl.Stage("Dissipation"):
            dstb_state_dependent = dstb_depends_on_state(my_object)

            # Reduce the per-row derivative bounds of the Hamiltonian stage (serial, N0 iterations)
            rows_min(min_deriv1_rows, min_deriv1, "r_min_deriv1")
            rows_min(min_deriv2_rows, min_deriv2, "r_min_deriv2")
            rows_min(min_deriv3_rows, min_deriv3, "r_min_deriv3")
            rows_min(min_deriv4_rows, min_deriv4, "r_min_deriv4")
            rows_max(max_deriv1_rows, max_deriv1, "r_max_deriv1")
            rows_max(max_deriv2_rows, max_deriv2, "r_max_deriv2")
            rows_max(max_deriv3_rows, max_deriv3, "r_max_deriv3")
            rows_max(max_deriv4_rows, max_deriv4, "r_max_deriv4")

            # Optimal disturbance at the LOWER/UPPER BOUND derivatives. If the dynamics sets
            # dstb_state_dependent = False, opt_dstb does not depend on the state and is evaluated once
            # here; otherwise it is evaluated at every grid point inside the loop below.
            if not dstb_state_dependent:
                dOptL1 = hcl.scalar(0, "dOptL1")
                dOptL2 = hcl.scalar(0, "dOptL2")
                dOptL3 = hcl.scalar(0, "dOptL3")
                dOptL4 = hcl.scalar(0, "dOptL4")
                dOptU1 = hcl.scalar(0, "dOptU1")
                dOptU2 = hcl.scalar(0, "dOptU2")
                dOptU3 = hcl.scalar(0, "dOptU3")
                dOptU4 = hcl.scalar(0, "dOptU4")
                dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0] = my_object.opt_dstb(t, (x1[0], x2[0], x3[0], x4[0]),
                                                                                (min_deriv1[0], min_deriv2[0], \
                                                                                min_deriv3[0], min_deriv4[0]))

                dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0] = my_object.opt_dstb(t, (x1[0], x2[0], x3[0], x4[0]),
                                                                                (max_deriv1[0], max_deriv2[0], \
                                                                                max_deriv3[0], max_deriv4[0]))

            with hcl.for_(0, V_init.shape[0], name="i") as i:
                # Per-point variables: declared once per outer index (inside the parallel loop, so private
                # to each thread) and reused for every point of that row
                alpha1 = hcl.scalar(0, "alpha1")
                alpha2 = hcl.scalar(0, "alpha2")
                alpha3 = hcl.scalar(0, "alpha3")
                alpha4 = hcl.scalar(0, "alpha4")
                uOptL1 = hcl.scalar(0, "uOptL1")
                uOptL2 = hcl.scalar(0, "uOptL2")
                uOptL3 = hcl.scalar(0, "uOptL3")
                uOptL4 = hcl.scalar(0, "uOptL4")
                uOptU1 = hcl.scalar(0, "uOptU1")
                uOptU2 = hcl.scalar(0, "uOptU2")
                uOptU3 = hcl.scalar(0, "uOptU3")
                uOptU4 = hcl.scalar(0, "uOptU4")
                if dstb_state_dependent:
                    dOptL1 = hcl.scalar(0, "dOptL1")
                    dOptL2 = hcl.scalar(0, "dOptL2")
                    dOptL3 = hcl.scalar(0, "dOptL3")
                    dOptL4 = hcl.scalar(0, "dOptL4")
                    dOptU1 = hcl.scalar(0, "dOptU1")
                    dOptU2 = hcl.scalar(0, "dOptU2")
                    dOptU3 = hcl.scalar(0, "dOptU3")
                    dOptU4 = hcl.scalar(0, "dOptU4")
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

                            dx_UU1 = hcl.scalar(0, "dx_UU1")
                            dx_UU2 = hcl.scalar(0, "dx_UU2")
                            dx_UU3 = hcl.scalar(0, "dx_UU3")
                            dx_UU4 = hcl.scalar(0, "dx_UU4")

                            dx_LU1 = hcl.scalar(0, "dx_LU1")
                            dx_LU2 = hcl.scalar(0, "dx_LU2")
                            dx_LU3 = hcl.scalar(0, "dx_LU3")
                            dx_LU4 = hcl.scalar(0, "dx_LU4")

                            if dstb_state_dependent:
                                # Find LOWER BOUND optimal disturbance
                                dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0] = my_object.opt_dstb(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                                (min_deriv1[0], min_deriv2[0], \
                                                                                min_deriv3[0], min_deriv4[0]))
                                # Find UPPER BOUND optimal disturbance
                                dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0] = my_object.opt_dstb(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                                (max_deriv1[0], max_deriv2[0], \
                                                                                max_deriv3[0], max_deriv4[0]))

                            # Find LOWER BOUND optimal control
                            uOptL1[0], uOptL2[0], uOptL3[0], uOptL4[0]= my_object.opt_ctrl(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                      (min_deriv1[0], min_deriv2[0], min_deriv3[0],
                                                                       min_deriv4[0]))

                            # Find UPPER BOUND optimal control
                            uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0] = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                      (max_deriv1[0], max_deriv2[0], max_deriv3[0],
                                                                       max_deriv4[0]))

                            # Find magnitude of rates of changes
                            dx_LL1[0], dx_LL2[0], dx_LL3[0], dx_LL4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                                            (uOptL1[0], uOptL2[0],uOptL3[0], uOptL4[0]),\
                                                                                            (dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0]))
                            dx_LL1[0] = my_abs(dx_LL1[0])
                            dx_LL2[0] = my_abs(dx_LL2[0])
                            dx_LL3[0] = my_abs(dx_LL3[0])
                            dx_LL4[0] = my_abs(dx_LL4[0])

                            dx_LU1[0], dx_LU2[0], dx_LU3[0], dx_LU4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                                            (uOptL1[0], uOptL2[0],uOptL3[0], uOptL4[0]), \
                                                                                            (dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0]))
                            dx_LU1[0] = my_abs(dx_LU1[0])
                            dx_LU2[0] = my_abs(dx_LU2[0])
                            dx_LU3[0] = my_abs(dx_LU3[0])
                            dx_LU4[0] = my_abs(dx_LU4[0])

                            # Calculate alpha
                            alpha1[0] = my_max(dx_LL1[0], dx_LU1[0])
                            alpha2[0] = my_max(dx_LL2[0], dx_LU2[0])
                            alpha3[0] = my_max(dx_LL3[0], dx_LU3[0])
                            alpha4[0] = my_max(dx_LL4[0], dx_LU4[0])

                            dx_UL1[0], dx_UL2[0], dx_UL3[0], dx_UL4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]),\
                                                                                            (uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0]), \
                                                                                            (dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0]))
                            dx_UL1[0] = my_abs(dx_UL1[0])
                            dx_UL2[0] = my_abs(dx_UL2[0])
                            dx_UL3[0] = my_abs(dx_UL3[0])
                            dx_UL4[0] = my_abs(dx_UL4[0])
                            
                            # Calculate alpha
                            alpha1[0] = my_max(alpha1[0], dx_UL1[0])
                            alpha2[0] = my_max(alpha2[0], dx_UL2[0])
                            alpha3[0] = my_max(alpha3[0], dx_UL3[0])
                            alpha4[0] = my_max(alpha4[0], dx_UL4[0])

                            dx_UU1[0], dx_UU2[0], dx_UU3[0], dx_UU4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                                            (uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0]),\
                                                                                            (dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0]))
                            dx_UU1[0] = my_abs(dx_UU1[0])
                            dx_UU2[0] = my_abs(dx_UU2[0])
                            dx_UU3[0] = my_abs(dx_UU3[0])
                            dx_UU4[0] = my_abs(dx_UU4[0])

                            # Calculate alphas
                            alpha1[0] = my_max(alpha1[0], dx_UU1[0])
                            alpha2[0] = my_max(alpha2[0], dx_UU2[0])
                            alpha3[0] = my_max(alpha3[0], dx_UU3[0])
                            alpha4[0] = my_max(alpha4[0], dx_UU4[0])

                            diss = hcl.scalar(0, "diss")
                            diss[0] = 0.5 * (
                                        deriv_diff1[i, j, k, l] * alpha1[0] + deriv_diff2[i, j, k, l] * alpha2[0] + deriv_diff3[
                                    i, j, k, l] * alpha3[0] + deriv_diff4[i, j, k, l] * alpha4[0])
                            #probe[i, j, k, l] = alpha1[0]

                            # Finally
                            V_new[i, j, k, l] = -(V_new[i, j, k, l] - diss[0])
                            
                            # Get maximum alphas in each dimension
                            update_row_max(max_alpha1_rows, i, alpha1[0])
                            update_row_max(max_alpha2_rows, i, alpha2[0])
                            update_row_max(max_alpha3_rows, i, alpha3[0])
                            update_row_max(max_alpha4_rows, i, alpha4[0])

            # Reduce the per-row alphas for the CFL time step (serial, N0 iterations)
            rows_max(max_alpha1_rows, max_alpha1, "r_max_alpha1")
            rows_max(max_alpha2_rows, max_alpha2, "r_max_alpha2")
            rows_max(max_alpha3_rows, max_alpha3, "r_max_alpha3")
            rows_max(max_alpha4_rows, max_alpha4, "r_max_alpha4")

        # Update largest time step - CFL condition
        hcl.update(delta_t, lambda x: step_bound())
        
    def returnDerivative(V_array, Deriv_array):
        with hcl.Stage("ComputeDeriv"):
            with hcl.for_(0, V_array.shape[0], name="i") as i:
                with hcl.for_(0, V_array.shape[1], name="j") as j:
                    with hcl.for_(0, V_array.shape[2], name="k") as k:
                        with hcl.for_(0, V_array.shape[3], name="l") as l:
                            dV_dx_L = hcl.scalar(0, "dV_dx_L")
                            dV_dx_R = hcl.scalar(0, "dV_dx_R")
                            if accuracy == "low":
                                if deriv_dim == 1:
                                    dV_dx_L[0], dV_dx_R[0] = spa_derivX1_4d(i, j, k, l, V_array, g)
                                if deriv_dim == 2:
                                    dV_dx_L[0], dV_dx_R[0] = spa_derivX2_4d(i, j, k, l, V_array, g)
                                if deriv_dim == 3:
                                    dV_dx_L[0], dV_dx_R[0] = spa_derivX3_4d(i, j, k, l, V_array, g)
                                if deriv_dim == 4:
                                    dV_dx_L[0], dV_dx_R[0] = spa_derivX4_4d(i, j, k, l, V_array, g)
                            if accuracy == "medium":
                                if deriv_dim == 1:
                                    dV_dx_L[0], dV_dx_R[0] = secondOrder_ENO4D_X0(i, j, k, l, V_array, g)
                                if deriv_dim == 2:
                                    dV_dx_L[0], dV_dx_R[0] = secondOrder_ENO4D_X1(i, j, k, l, V_array, g)
                                if deriv_dim == 3:
                                    dV_dx_L[0], dV_dx_R[0] = secondOrder_ENO4D_X2(i, j, k, l, V_array, g)
                                if deriv_dim == 4:
                                    dV_dx_L[0], dV_dx_R[0] = secondOrder_ENO4D_X3(i, j, k, l, V_array, g)

                            Deriv_array[i, j, k, l] = (dV_dx_L[0] + dV_dx_R[0]) / 2

    if generate_SpatDeriv == False:
        s = hcl.create_schedule([V_f, V_init, x1, x2, x3, x4, delta_t, t, l0], graph_create)

        ##################### CODE OPTIMIZATION HERE ###########################
        print("Optimizing\n")

        # Accessing the hamiltonian and dissipation stage
        s_H = graph_create.Hamiltonian
        s_D = graph_create.Dissipation

        # Thread parallelize hamiltonian and dissipation
        s[s_H].parallel(s_H.i)
        s[s_D].parallel(s_D.i)

        # Inspect IR
        # if args.llvm:
        #    print(hcl.lower(s))
    else:
        s = hcl.create_schedule([V_init, V_f], returnDerivative)

    # Return executable
    return(hcl.build(s))
