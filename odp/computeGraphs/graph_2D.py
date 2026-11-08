import heterocl as hcl
import numpy as np
from odp.computeGraphs.CustomGraphFunctions import *
from odp.spatialDerivatives.firstOrderENO.first_orderENO2D import *
from odp.spatialDerivatives.secondOrderENO.second_orderENO2D import *

#from user_definer import *
#def graph_2D(dynamics_obj, grid):
def graph_2D(my_object, g, compMethod, accuracy, generate_SpatDeriv=False, deriv_dim=1):
    V_f = hcl.placeholder(tuple(g.pts_each_dim), name="V_f", dtype=hcl.Float())
    V_init = hcl.placeholder(tuple(g.pts_each_dim), name="V_init", dtype=hcl.Float())
    l0 = hcl.placeholder(tuple(g.pts_each_dim), name="l0", dtype=hcl.Float())
    t = hcl.placeholder((1,), name="t", dtype=hcl.Float())
    delta_t = hcl.placeholder((1,), name="delta_t", dtype=hcl.Float())
    probe = hcl.placeholder(tuple(g.pts_each_dim), name="probe", dtype=hcl.Float())

    # Positions vector
    x1 = hcl.placeholder((g.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((g.pts_each_dim[1],), name="x2", dtype=hcl.Float())

    def graph_create(V_new, V_init, x1, x2, delta_t, t, l0):
        # Specify intermediate tensors
        deriv_diff1 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff1")
        deriv_diff2 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff2")

        # Maximum derivative for each dim
        max_deriv1 = hcl.scalar(-1e9, "max_deriv1")
        max_deriv2 = hcl.scalar(-1e9, "max_deriv2")

        # Min derivative for each dim
        min_deriv1 = hcl.scalar(1e9, "min_deriv1")
        min_deriv2 = hcl.scalar(1e9, "min_deriv2")

        # These variables are used to dissipation calculation
        max_alpha1 = hcl.scalar(-1e9, "max_alpha1")
        max_alpha2 = hcl.scalar(-1e9, "max_alpha2")

        # Per-row (outermost index) running max/min, so threads of the parallel loops never write
        # the same location; reduced serially into the scalars above
        N0 = V_init.shape[0]
        max_deriv1_rows = hcl.compute((N0,), lambda x: -1e9, "max_deriv1_rows")
        max_deriv2_rows = hcl.compute((N0,), lambda x: -1e9, "max_deriv2_rows")
        max_alpha1_rows = hcl.compute((N0,), lambda x: -1e9, "max_alpha1_rows")
        max_alpha2_rows = hcl.compute((N0,), lambda x: -1e9, "max_alpha2_rows")
        min_deriv1_rows = hcl.compute((N0,), lambda x: 1e9, "min_deriv1_rows")
        min_deriv2_rows = hcl.compute((N0,), lambda x: 1e9, "min_deriv2_rows")

        def step_bound():  # Function to calculate time step
            stepBoundInv = hcl.scalar(0, "stepBoundInv")
            stepBound = hcl.scalar(0, "stepBound")
            stepBoundInv[0] = max_alpha1[0] / g.dx[0] + max_alpha2[0] / g.dx[1] 
            stepBound[0] = 0.8 / stepBoundInv[0]
            return stepBound[0]

        # Calculate Hamiltonian for every grid point in V_init
        with hcl.Stage("Hamiltonian"):
            with hcl.for_(0, V_init.shape[0], name="i") as i:  # Plus 1 as for loop count stops at V_init.shape[0]
                with hcl.for_(0, V_init.shape[1], name="j") as j:
                    # Variables to calculate dV_dx
                    dV_dx_L = hcl.scalar(0, "dV_dx_L")
                    dV_dx_R = hcl.scalar(0, "dV_dx_R")
                    dV_dx = hcl.scalar(0, "dV_dx")
                    # Variables to calculate dV_dy
                    dV_dy_L = hcl.scalar(0, "dV_dy_L")
                    dV_dy_R = hcl.scalar(0, "dV_dy_R")
                    dV_dy = hcl.scalar(0, "dV_dy")

                    # No tensor slice operation
                    if accuracy == "low":
                        dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, V_init, g)
                        dV_dy_L[0], dV_dy_R[0] = spa_derivY(i, j, V_init, g)
                    if accuracy == "medium":
                        dV_dx_L[0], dV_dx_R[0] = secondOrder_ENO2D_X0(i, j, V_init, g)
                        dV_dy_L[0], dV_dy_R[0] = secondOrder_ENO2D_X1(i, j, V_init, g)

                    # Saves spatial derivative diff into tables
                    deriv_diff1[i, j] = dV_dx_R[0] - dV_dx_L[0]
                    deriv_diff2[i, j] = dV_dy_R[0] - dV_dy_L[0]

                    # Calculate average gradient
                    dV_dx[0] = (dV_dx_L + dV_dx_R) / 2
                    dV_dy[0] = (dV_dy_L + dV_dy_R) / 2

                    # Use method of DubinsCar to solve optimal control instead
                    uOpt = my_object.opt_ctrl(t, (x1[i], x2[j]),
                                                    (dV_dx[0], dV_dy[0]))
                    dOpt = my_object.opt_dstb(t, (x1[i], x2[j]),
                                                (dV_dx[0], dV_dy[0]))

                    # Calculate dynamical rates of changes
                    dx_dt, dy_dt = my_object.dynamics(t, (x1[i], x2[j]), uOpt, dOpt)

                    # Calculate Hamiltonian terms:
                    V_new[i, j] = -(dx_dt * dV_dx[0] + dy_dt * dV_dy[0])

                    # Get derivMin
                    update_row_min(min_deriv1_rows, i, dV_dx_L[0])
                    update_row_min(min_deriv1_rows, i, dV_dx_R[0])

                    update_row_min(min_deriv2_rows, i, dV_dy_L[0])
                    update_row_min(min_deriv2_rows, i, dV_dy_R[0])

                    # Get derivMax
                    update_row_max(max_deriv1_rows, i, dV_dx_L[0])
                    update_row_max(max_deriv1_rows, i, dV_dx_R[0])

                    update_row_max(max_deriv2_rows, i, dV_dy_L[0])
                    update_row_max(max_deriv2_rows, i, dV_dy_R[0])


        # Calculate the dissipation
        with hcl.Stage("Dissipation"):
            dstb_state_dependent = dstb_depends_on_state(my_object)

            # Reduce the per-row derivative bounds of the Hamiltonian stage (serial, N0 iterations)
            rows_min(min_deriv1_rows, min_deriv1, "r_min_deriv1")
            rows_min(min_deriv2_rows, min_deriv2, "r_min_deriv2")
            rows_max(max_deriv1_rows, max_deriv1, "r_max_deriv1")
            rows_max(max_deriv2_rows, max_deriv2, "r_max_deriv2")

            # Optimal disturbance at the LOWER/UPPER BOUND derivatives. If the dynamics sets
            # dstb_state_dependent = False, opt_dstb does not depend on the state and is evaluated once
            # here; otherwise it is evaluated at every grid point inside the loop below.
            if not dstb_state_dependent:
                dOptL1 = hcl.scalar(0, "dOptL1")
                dOptL2 = hcl.scalar(0, "dOptL2")
                dOptU1 = hcl.scalar(0, "dOptU1")
                dOptU2 = hcl.scalar(0, "dOptU2")
                # Find LOWER BOUND optimal disturbance
                dOptL1[0], dOptL2[0] = my_object.opt_dstb(t,  (x1[0], x2[0]),\
                                                                    (min_deriv1[0], min_deriv2[0]))

                dOptU1[0], dOptU2[0] = my_object.opt_dstb(t, (x1[0], x2[0]),\
                                                                    (max_deriv1[0], max_deriv2[0]))

            with hcl.for_(0, V_init.shape[0], name="i") as i:
                # Per-point variables: declared once per outer index (inside the parallel loop, so private
                # to each thread) and reused for every point of that row
                alpha1 = hcl.scalar(0, "alpha1")
                alpha2 = hcl.scalar(0, "alpha2")
                uOptL1 = hcl.scalar(0, "uOptL1")
                uOptL2 = hcl.scalar(0, "uOptL2")
                uOptU1 = hcl.scalar(0, "uOptU1")
                uOptU2 = hcl.scalar(0, "uOptU2")
                if dstb_state_dependent:
                    dOptL1 = hcl.scalar(0, "dOptL1")
                    dOptL2 = hcl.scalar(0, "dOptL2")
                    dOptU1 = hcl.scalar(0, "dOptU1")
                    dOptU2 = hcl.scalar(0, "dOptU2")
                with hcl.for_(0, V_init.shape[1], name="j") as j:
                    dx_LL1 = hcl.scalar(0, "dx_LL1")
                    dx_LL2 = hcl.scalar(0, "dx_LL2")

                    dx_UL1 = hcl.scalar(0, "dx_UL1")
                    dx_UL2 = hcl.scalar(0, "dx_UL2")

                    dx_UU1 = hcl.scalar(0, "dx_UU1")
                    dx_UU2 = hcl.scalar(0, "dx_UU2")

                    dx_LU1 = hcl.scalar(0, "dx_LU1")
                    dx_LU2 = hcl.scalar(0, "dx_LU2")

                    if dstb_state_dependent:
                        # Find LOWER BOUND optimal disturbance
                        dOptL1[0], dOptL2[0] = my_object.opt_dstb(t,  (x1[i], x2[j]),\
                                                                            (min_deriv1[0], min_deriv2[0]))

                        dOptU1[0], dOptU2[0] = my_object.opt_dstb(t, (x1[i], x2[j]),\
                                                                            (max_deriv1[0], max_deriv2[0]))

                    # Find LOWER BOUND optimal control
                    uOptL1[0], uOptL2[0] = my_object.opt_ctrl(t, (x1[i], x2[j]), \
                                                                                    (min_deriv1[0], min_deriv2[0]))

                        # Find UPPER BOUND optimal control
                    uOptU1[0], uOptU2[0] = my_object.opt_ctrl(t, (x1[i], x2[j]),
                                                                                        (max_deriv1[0], max_deriv2[0]))
                        # Find magnitude of rates of changes
                    dx_LL1[0], dx_LL2[0] = my_object.dynamics(t, (x1[i], x2[j]),
                                                                                        (uOptL1[0], uOptL2[0]), \
                                                                                        (dOptL1[0], dOptL2[0]))
                    dx_LL1[0] = my_abs(dx_LL1[0])
                    dx_LL2[0] = my_abs(dx_LL2[0])

                    dx_LU1[0], dx_LU2[0] = my_object.dynamics(t, (x1[i], x2[j]),
                                                                                    (uOptL1[0], uOptL2[0]), \
                                                                                    (dOptU1[0], dOptU2[0]))
                    dx_LU1[0] = my_abs(dx_LU1[0])
                    dx_LU2[0] = my_abs(dx_LU2[0])

                    # Calculate alpha
                    alpha1[0] = my_max(dx_LL1[0], dx_LU1[0])
                    alpha2[0] = my_max(dx_LL2[0], dx_LU2[0])

                    dx_UL1[0], dx_UL2[0] = my_object.dynamics(t, (x1[i], x2[j]),\
                                                                                        (uOptU1[0], uOptU2[0]), \
                                                                                        (dOptL1[0], dOptL2[0]))
                    dx_UL1[0] = my_abs(dx_UL1[0])
                    dx_UL2[0] = my_abs(dx_UL2[0])

                    # Calculate alpha
                    alpha1[0] = my_max(alpha1[0], dx_UL1[0])
                    alpha2[0] = my_max(alpha2[0], dx_UL2[0])

                    dx_UU1[0], dx_UU2[0] = my_object.dynamics(t, (x1[i], x2[j]),
                                                                                        (uOptU1[0], uOptU2[0]),\
                                                                                        (dOptU1[0], dOptU2[0]))
                    dx_UU1[0] = my_abs(dx_UU1[0])
                    dx_UU2[0] = my_abs(dx_UU2[0])
                    # Calculate alpha
                    alpha1[0] = my_max(alpha1[0], dx_UU1[0])
                    alpha2[0] = my_max(alpha2[0], dx_UU2[0])

                    diss = hcl.scalar(0, "diss")
                    diss[0] = 0.5 * (
                            deriv_diff1[i, j] * alpha1[0] + deriv_diff2[i, j] * alpha2[0])

                    # Finally
                    V_new[i, j] = -(V_new[i, j] - diss[0])

                    # Calculate alphas
                    update_row_max(max_alpha1_rows, i, alpha1[0])
                    update_row_max(max_alpha2_rows, i, alpha2[0])

            # Reduce the per-row alphas for the CFL time step (serial, N0 iterations)
            rows_max(max_alpha1_rows, max_alpha1, "r_max_alpha1")
            rows_max(max_alpha2_rows, max_alpha2, "r_max_alpha2")

        # Update largest time step - CFL condition
        hcl.update(delta_t, lambda x: step_bound())

    def returnDerivative(V_array, Deriv_array):
        with hcl.Stage("ComputeDeriv"):
            with hcl.for_(0, V_array.shape[0], name="i") as i:
                with hcl.for_(0, V_array.shape[1], name="j") as j:
                    dV_dx_L = hcl.scalar(0, "dV_dx_L")
                    dV_dx_R = hcl.scalar(0, "dV_dx_R")
                    if accuracy == "low":
                        if deriv_dim == 1:
                            dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, V_array, g)
                        if deriv_dim == 2:
                            dV_dx_L[0], dV_dx_R[0] = spa_derivY(i, j, V_array, g)
                    if accuracy == "medium":
                        if deriv_dim == 1:
                            dV_dx_L[0], dV_dx_R[0] = secondOrder_ENO2D_X0(i, j, V_array, g)
                        if deriv_dim == 2:
                            dV_dx_L[0], dV_dx_R[0] = secondOrder_ENO2D_X1(i, j, V_array, g)

                    Deriv_array[i, j] = (dV_dx_L[0] + dV_dx_R[0]) / 2

    if generate_SpatDeriv == False:
        s = hcl.create_schedule([V_f, V_init, x1, x2, delta_t, t, l0], graph_create)
        ##################### CODE OPTIMIZATION HERE ###########################
        print("Optimizing\n")

        # Accessing the hamiltonian and dissipation stage
        s_H = graph_create.Hamiltonian
        s_D = graph_create.Dissipation

        # Thread parallelize hamiltonian and dissipation computation
        s[s_H].parallel(s_H.i)
        s[s_D].parallel(s_D.i)
    else:
        print("I'm here\n")
        s = hcl.create_schedule([V_init, V_f], returnDerivative)

    # Inspect IR
    # if args.llvm:
        #print(hcl.lower(s))

    # Return executable
    return (hcl.build(s))
