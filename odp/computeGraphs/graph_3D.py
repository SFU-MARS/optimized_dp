import heterocl as hcl
import numpy as np
from odp.computeGraphs.CustomGraphFunctions import *
from odp.spatialDerivatives.first_orderENO3D import *
from odp.spatialDerivatives.second_orderENO3D import *

#from user_definer import *
#def graph_3D(dynamics_obj, grid):
def graph_3D(my_object, g, compMethod, accuracy, generate_SpatDeriv=False, deriv_dim=1):
    V_f = hcl.placeholder(tuple(g.pts_each_dim), name="V_f", dtype=hcl.Float())
    V_init = hcl.placeholder(tuple(g.pts_each_dim), name="V_init", dtype=hcl.Float())
    l0 = hcl.placeholder(tuple(g.pts_each_dim), name="l0", dtype=hcl.Float())
    t = hcl.placeholder((2,), name="t", dtype=hcl.Float())
    probe = hcl.placeholder(tuple(g.pts_each_dim), name="probe", dtype=hcl.Float())

    # Positions vector
    x1 = hcl.placeholder((g.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((g.pts_each_dim[1],), name="x2", dtype=hcl.Float())
    x3 = hcl.placeholder((g.pts_each_dim[2],), name="x3", dtype=hcl.Float())
    def graph_create(V_new, V_init, x1, x2, x3, t, l0):
        # Specify intermediate tensors
        deriv_diff1 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff1")
        deriv_diff2 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff2")
        deriv_diff3 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff3")

        # Maximum derivative for each dim
        max_deriv1 = hcl.scalar(-1e9, "max_deriv1")
        max_deriv2 = hcl.scalar(-1e9, "max_deriv2")
        max_deriv3 = hcl.scalar(-1e9, "max_deriv3")

        # Min derivative for each dim
        min_deriv1 = hcl.scalar(1e9, "min_deriv1")
        min_deriv2 = hcl.scalar(1e9, "min_deriv2")
        min_deriv3 = hcl.scalar(1e9, "min_deriv3")

        # These variables are used to dissipation calculation
        max_alpha1 = hcl.scalar(-1e9, "max_alpha1")
        max_alpha2 = hcl.scalar(-1e9, "max_alpha2")
        max_alpha3 = hcl.scalar(-1e9, "max_alpha3")
        def step_bound():  # Function to calculate time step
            stepBoundInv = hcl.scalar(0, "stepBoundInv")
            stepBound = hcl.scalar(0, "stepBound")
            stepBoundInv[0] = max_alpha1[0] / g.dx[0] + max_alpha2[0] / g.dx[1] + max_alpha3[0] / g.dx[2]
            stepBound[0] = 0.8 / stepBoundInv[0]
            with hcl.if_(stepBound > t[1] - t[0]):
                stepBound[0] = t[1] - t[0]
            t[0] = t[0] + stepBound[0]
            return stepBound[0]

            # Min with V_before
        def minVWithVInit(i, j, k):
            with hcl.if_(V_new[i, j, k] > V_init[i, j, k]):
                V_new[i, j, k] = V_init[i, j, k]

        def maxVWithVInit(i, j, k):
            with hcl.if_(V_new[i, j, k] < V_init[i, j, k]):
                V_new[i, j, k] = V_init[i, j, k]

        def maxVWithV0(i, j, k):  # Take the max
            with hcl.if_(V_new[i, j, k] < l0[i, j, k]):
                V_new[i, j, k] = l0[i, j, k]

        def minVWithV0(i, j, k):
            with hcl.if_(V_new[i, j, k] > l0[i, j, k]):
                V_new[i, j, k] = l0[i, j, k]

        # Calculate Hamiltonian for every grid point in V_init
        with hcl.Stage("Hamiltonian"):
            with hcl.for_(0, V_init.shape[0], name="i") as i:  # Plus 1 as for loop count stops at V_init.shape[0]
                with hcl.for_(0, V_init.shape[1], name="j") as j:
                    with hcl.for_(0, V_init.shape[2], name="k") as k:
                        # Variables to calculate dV_dx
                        dV_dx_L = hcl.scalar(0, "dV_dx_L")
                        dV_dx_R = hcl.scalar(0, "dV_dx_R")
                        dV_dx = hcl.scalar(0, "dV_dx")
                        # Variables to calculate dV_dy
                        dV_dy_L = hcl.scalar(0, "dV_dy_L")
                        dV_dy_R = hcl.scalar(0, "dV_dy_R")
                        dV_dy = hcl.scalar(0, "dV_dy")
                        # Variables to calculate dV_dtheta
                        dV_dT_L = hcl.scalar(0, "dV_dT_L")
                        dV_dT_R = hcl.scalar(0, "dV_dT_R")
                        dV_dT = hcl.scalar(0, "dV_dT")

                        # No tensor slice operation
                        if accuracy == "low":
                            dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k, V_init, g)
                            dV_dy_L[0], dV_dy_R[0] = spa_derivY(i, j, k, V_init, g)
                            dV_dT_L[0], dV_dT_R[0] = spa_derivT(i, j, k, V_init, g)
                        if accuracy == "medium":
                            dV_dx_L[0], dV_dx_R[0] = secondOrderX(i, j, k, V_init, g)
                            dV_dy_L[0], dV_dy_R[0] = secondOrderY(i, j, k, V_init, g)
                            dV_dT_L[0], dV_dT_R[0] = secondOrderT(i, j, k, V_init, g)

                        # Saves spatial derivative diff into tables
                        deriv_diff1[i, j, k] = dV_dx_R[0] - dV_dx_L[0]
                        deriv_diff2[i, j, k] = dV_dy_R[0] - dV_dy_L[0]
                        deriv_diff3[i, j, k] = dV_dT_R[0] - dV_dT_L[0]

                        # Calculate average gradient
                        dV_dx[0] = (dV_dx_L + dV_dx_R) / 2
                        dV_dy[0] = (dV_dy_L + dV_dy_R) / 2
                        dV_dT[0] = (dV_dT_L + dV_dT_R) / 2

                        # Use method of DubinsCar to solve optimal control instead
                        uOpt = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k]),
                                                      (dV_dx[0], dV_dy[0], dV_dT[0]))
                        dOpt = my_object.opt_dstb(t, (x1[i], x2[j], x3[k]),
                                                 (dV_dx[0], dV_dy[0], dV_dT[0]))

                        # Calculate dynamical rates of changes
                        dx_dt, dy_dt, dtheta_dt = my_object.dynamics(t, (x1[i], x2[j], x3[k]), uOpt, dOpt)

                        # Calculate Hamiltonian terms:
                        V_new[i, j, k] = -(dx_dt * dV_dx[0] + dy_dt * dV_dy[0] + dtheta_dt * dV_dT[0])

                        # Get derivMin
                        with hcl.if_(dV_dx_L[0] < min_deriv1[0]):
                            min_deriv1[0] = dV_dx_L[0]
                        with hcl.if_(dV_dx_R[0] < min_deriv1[0]):
                            min_deriv1[0] = dV_dx_R[0]

                        with hcl.if_(dV_dy_L[0] < min_deriv2[0]):
                            min_deriv2[0] = dV_dy_L[0]
                        with hcl.if_(dV_dy_R[0] < min_deriv2[0]):
                            min_deriv2[0] = dV_dy_R[0]

                        with hcl.if_(dV_dT_L[0] < min_deriv3[0]):
                            min_deriv3[0] = dV_dT_L[0]
                        with hcl.if_(dV_dT_R[0] < min_deriv3[0]):
                            min_deriv3[0] = dV_dT_R[0]

                        # Get derivMax
                        with hcl.if_(dV_dx_L[0] > max_deriv1[0]):
                            max_deriv1[0] = dV_dx_L[0]
                        with hcl.if_(dV_dx_R[0] > max_deriv1[0]):
                            max_deriv1[0] = dV_dx_R[0]

                        with hcl.if_(dV_dy_L[0] > max_deriv2[0]):
                            max_deriv2[0] = dV_dy_L[0]
                        with hcl.if_(dV_dy_R[0] > max_deriv2[0]):
                            max_deriv2[0] = dV_dy_R[0]

                        with hcl.if_(dV_dT_L[0] > max_deriv3[0]):
                            max_deriv3[0] = dV_dT_L[0]
                        with hcl.if_(dV_dT_R[0] > max_deriv3[0]):
                            max_deriv3[0] = dV_dT_R[0]

        # Calculate the dissipation
        with hcl.Stage("Dissipation"):
            # Storing alphas
            dOptL1 = hcl.scalar(0, "dOptL1")
            dOptL2 = hcl.scalar(0, "dOptL2")
            dOptL3 = hcl.scalar(0, "dOptL3")
            # Find UPPER BOUND optimal disturbance
            dOptU1 = hcl.scalar(0, "dOptU1")
            dOptU2 = hcl.scalar(0, "dOptU2")
            dOptU3 = hcl.scalar(0, "dOptU3")

            alpha1 = hcl.scalar(0, "alpha1")
            alpha2 = hcl.scalar(0, "alpha2")
            alpha3 = hcl.scalar(0, "alpha3")

            # Lower bound optimal control
            uOptL1 = hcl.scalar(0, "uOptL1")
            uOptL2 = hcl.scalar(0, "uOptL2")
            uOptL3 = hcl.scalar(0, "uOptL3")

            # Find UPPER BOUND optimal disturbance
            uOptU1 = hcl.scalar(0, "uOptU1")
            uOptU2 = hcl.scalar(0, "uOptU2")
            uOptU3 = hcl.scalar(0, "uOptU3")

            with hcl.for_(0, V_init.shape[0], name="i") as i:
                with hcl.for_(0, V_init.shape[1], name="j") as j:
                    with hcl.for_(0, V_init.shape[2], name="k") as k:
                        dx_LL1 = hcl.scalar(0, "dx_LL1")
                        dx_LL2 = hcl.scalar(0, "dx_LL2")
                        dx_LL3 = hcl.scalar(0, "dx_LL3")

                        dx_UL1 = hcl.scalar(0, "dx_UL1")
                        dx_UL2 = hcl.scalar(0, "dx_UL2")
                        dx_UL3 = hcl.scalar(0, "dx_UL3")

                        dx_UU1 = hcl.scalar(0, "dx_UU1")
                        dx_UU2 = hcl.scalar(0, "dx_UU2")
                        dx_UU3 = hcl.scalar(0, "dx_UU3")

                        dx_LU1 = hcl.scalar(0, "dx_LU1")
                        dx_LU2 = hcl.scalar(0, "dx_LU2")
                        dx_LU3 = hcl.scalar(0, "dx_LU3")

                        # Find LOWER BOUND optimal disturbance
                        dOptL1[0], dOptL2[0], dOptL3[0] = my_object.opt_dstb(t,  (x1[i], x2[j], x3[k]),\
                                                                            (min_deriv1[0], min_deriv2[0],min_deriv3[0]))

                        dOptU1[0], dOptU2[0], dOptU3[0] = my_object.opt_dstb(t, (x1[i], x2[j], x3[k]),\
                                                                            (max_deriv1[0], max_deriv2[0],max_deriv3[0]))

                        # Find LOWER BOUND optimal control
                        uOptL1[0], uOptL2[0], uOptL3[0] = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k]), \
                                                                                        (min_deriv1[0], min_deriv2[0],min_deriv3[0]))

                        # Find UPPER BOUND optimal control
                        uOptU1[0], uOptU2[0], uOptU3[0] = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k]),
                                                                                        (max_deriv1[0], max_deriv2[0],
                                                                                         max_deriv3[0]))
                        # Find magnitude of rates of changes
                        dx_LL1[0], dx_LL2[0], dx_LL3[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k]),
                                                                                        (uOptL1[0], uOptL2[0], uOptL3[0]), \
                                                                                        (dOptL1[0], dOptL2[0], dOptL3[0]))
                        dx_LL1[0] = my_abs(dx_LL1[0])
                        dx_LL2[0] = my_abs(dx_LL2[0])
                        dx_LL3[0] = my_abs(dx_LL3[0])

                        dx_LU1[0], dx_LU2[0], dx_LU3[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k]),
                                                                                        (uOptL1[0], uOptL2[0], uOptL3[0]), \
                                                                                        (dOptU1[0], dOptU2[0], dOptU3[0]))
                        dx_LU1[0] = my_abs(dx_LU1[0])
                        dx_LU2[0] = my_abs(dx_LU2[0])
                        dx_LU3[0] = my_abs(dx_LU3[0])

                        # Calculate alpha
                        alpha1[0] = my_max(dx_LL1[0], dx_LU1[0])
                        alpha2[0] = my_max(dx_LL2[0], dx_LU2[0])
                        alpha3[0] = my_max(dx_LL3[0], dx_LU3[0])

                        dx_UL1[0], dx_UL2[0], dx_UL3[0]= my_object.dynamics(t, (x1[i], x2[j], x3[k]),\
                                                                                            (uOptU1[0], uOptU2[0], uOptU3[0]), \
                                                                                            (dOptL1[0], dOptL2[0], dOptL3[0]))
                        dx_UL1[0] = my_abs(dx_UL1[0])
                        dx_UL2[0] = my_abs(dx_UL2[0])
                        dx_UL3[0] = my_abs(dx_UL3[0])

                        # Calculate alpha
                        alpha1[0] = my_max(alpha1[0], dx_UL1[0])
                        alpha2[0] = my_max(alpha2[0], dx_UL2[0])
                        alpha3[0] = my_max(alpha3[0], dx_UL3[0])

                        dx_UU1[0], dx_UU2[0], dx_UU3[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k]),
                                                                                            (uOptU1[0], uOptU2[0], uOptU3[0]),\
                                                                                            (dOptU1[0], dOptU2[0], dOptU3[0]))
                        dx_UU1[0] = my_abs(dx_UU1[0])
                        dx_UU2[0] = my_abs(dx_UU2[0])
                        dx_UU3[0] = my_abs(dx_UU3[0])
                        # Calculate alpha
                        alpha1[0] = my_max(alpha1[0], dx_UU1[0])
                        alpha2[0] = my_max(alpha2[0], dx_UU2[0])
                        alpha3[0] = my_max(alpha3[0], dx_UU3[0])

                        diss = hcl.scalar(0, "diss")
                        diss[0] = 0.5 * (
                                deriv_diff1[i, j, k] * alpha1[0] + deriv_diff2[i, j, k] * alpha2[0] + deriv_diff3[i, j, k] * alpha3[0])

                        # Finally
                        V_new[i, j, k] = -(V_new[i, j, k] - diss[0])

                        # Calculate alphas
                        with hcl.if_(alpha1[0] > max_alpha1[0]):
                            max_alpha1[0] = alpha1[0]
                        with hcl.if_(alpha2[0] > max_alpha2[0]):
                            max_alpha2[0] = alpha2[0]
                        with hcl.if_(alpha3[0] > max_alpha3[0]):
                            max_alpha3[0] = alpha3[0]



        # Determine time step
        delta_t = hcl.compute((1,), lambda x: step_bound(), name="delta_t")
        # Integrate
        result = hcl.update(V_new, lambda i, j, k: V_init[i, j, k] + V_new[i, j, k] * delta_t[0])

        # Different computation method check
        if compMethod == 'maxVWithV0' or compMethod == 'maxVWithVTarget':
            result = hcl.update(V_new, lambda i, j, k: maxVWithV0(i, j, k))
        if compMethod == 'minVWithV0' or compMethod == 'minVWithVTarget':
            result = hcl.update(V_new, lambda i, j, k: minVWithV0(i, j, k))
        if compMethod == 'minVWithVInit':
            result = hcl.update(V_new, lambda i, j, k: minVWithVInit(i, j, k))
        if compMethod == 'maxVWithVInit':
            result = hcl.update(V_new, lambda i, j, k: maxVWithVInit(i, j, k))

        # Copy V_new to V_init
        hcl.update(V_init, lambda i, j, k: V_new[i, j, k])
        return result

    def returnDerivative(V_array, Deriv_array):
        with hcl.Stage("ComputeDeriv"):
            with hcl.for_(0, V_array.shape[0], name="i") as i:
                with hcl.for_(0, V_array.shape[1], name="j") as j:
                    with hcl.for_(0, V_array.shape[2], name="k") as k:
                        dV_dx_L = hcl.scalar(0, "dV_dx_L")
                        dV_dx_R = hcl.scalar(0, "dV_dx_R")
                        if accuracy == "low":
                            if deriv_dim == 1:
                                dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k, V_array, g)
                            if deriv_dim == 2:
                                dV_dx_L[0], dV_dx_R[0] = spa_derivY(i, j, k, V_array, g)
                            if deriv_dim == 3:
                                dV_dx_L[0], dV_dx_R[0] = spa_derivT(i, j, k, V_array, g)
                        if accuracy == "medium":
                            if deriv_dim == 1:
                                dV_dx_L[0], dV_dx_R[0] = secondOrderX(i, j, k, V_array, g)
                            if deriv_dim == 2:
                                dV_dx_L[0], dV_dx_R[0] = secondOrderY(i, j, k, V_array, g)
                            if deriv_dim == 3:
                                dV_dx_L[0], dV_dx_R[0] = secondOrderT(i, j, k, V_array, g)

                        Deriv_array[i, j, k] = (dV_dx_L[0] + dV_dx_R[0]) / 2

    if generate_SpatDeriv == False:
        s = hcl.create_schedule([V_f, V_init, x1, x2, x3, t, l0], graph_create)
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
