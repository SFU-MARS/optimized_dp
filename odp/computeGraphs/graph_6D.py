import heterocl as hcl
from odp.computeGraphs.CustomGraphFunctions import *
from odp.spatialDerivatives.first_orderENO6D import *
from odp.spatialDerivatives.second_orderENO6D import *

########################## 6D graph definition ########################

# Note that t has 2 elements t1, t2
def graph_6D(my_object, g, compMethod, accuracy):
    V_f = hcl.placeholder(tuple(g.pts_each_dim), name="V_f", dtype=hcl.Float())
    V_init = hcl.placeholder(tuple(g.pts_each_dim), name="V_init", dtype=hcl.Float())
    l0 = hcl.placeholder(tuple(g.pts_each_dim), name="l0", dtype=hcl.Float())
    t = hcl.placeholder((2,), name="t", dtype=hcl.Float())

    # Positions vector
    x1 = hcl.placeholder((g.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((g.pts_each_dim[1],), name="x2", dtype=hcl.Float())
    x3 = hcl.placeholder((g.pts_each_dim[2],), name="x3", dtype=hcl.Float())
    x4 = hcl.placeholder((g.pts_each_dim[3],), name="x4", dtype=hcl.Float())
    x5 = hcl.placeholder((g.pts_each_dim[4],), name="x5", dtype=hcl.Float())
    x6 = hcl.placeholder((g.pts_each_dim[5],), name="x6", dtype=hcl.Float())

    def graph_create(V_new, V_init, x1, x2, x3, x4, x5, x6, t, l0):
        # Specify intermediate tensors
        deriv_diff1 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff1")
        deriv_diff2 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff2")
        deriv_diff3 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff3")
        deriv_diff4 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff4")
        deriv_diff5 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff5")
        deriv_diff6 = hcl.compute(V_init.shape, lambda *x: 0, "deriv_diff6")

        # Maximum derivative for each dim
        max_deriv1 = hcl.scalar(-1e9, "max_deriv1")
        max_deriv2 = hcl.scalar(-1e9, "max_deriv2")
        max_deriv3 = hcl.scalar(-1e9, "max_deriv3")
        max_deriv4 = hcl.scalar(-1e9, "max_deriv4")
        max_deriv5 = hcl.scalar(-1e9, "max_deriv5")
        max_deriv6 = hcl.scalar(-1e9, "max_deriv6")

        # Min derivative for each dim
        min_deriv1 = hcl.scalar(1e9, "min_deriv1")
        min_deriv2 = hcl.scalar(1e9, "min_deriv2")
        min_deriv3 = hcl.scalar(1e9, "min_deriv3")
        min_deriv4 = hcl.scalar(1e9, "min_deriv4")
        min_deriv5 = hcl.scalar(1e9, "min_deriv5")
        min_deriv6 = hcl.scalar(1e9, "min_deriv6")

        # These variables are used to dissipation calculation
        max_alpha1 = hcl.scalar(-1e9, "max_alpha1")
        max_alpha2 = hcl.scalar(-1e9, "max_alpha2")
        max_alpha3 = hcl.scalar(-1e9, "max_alpha3")
        max_alpha4 = hcl.scalar(-1e9, "max_alpha4")
        max_alpha5 = hcl.scalar(-1e9, "max_alpha5")
        max_alpha6 = hcl.scalar(-1e9, "max_alpha6")

        def step_bound():  # Function to calculate time step
            stepBoundInv = hcl.scalar(0, "stepBoundInv")
            stepBound = hcl.scalar(0, "stepBound")
            stepBoundInv[0] = max_alpha1[0] / g.dx[0] + max_alpha2[0] / g.dx[1] + max_alpha3[0] / g.dx[2] + max_alpha4[
                0] / g.dx[3] \
                              + max_alpha5[0] / g.dx[4] + max_alpha6[0] / g.dx[5]

            stepBound[0] = 0.8 / stepBoundInv[0]
            with hcl.if_(stepBound > t[1] - t[0]):
                stepBound[0] = t[1] - t[0]

            # Update the lower time ranges
            t[0] = t[0] + stepBound[0]
            # t[0] = min_deriv2[0]
            return stepBound[0]

        # Operation with target value array
        def maxVWithV0(i, j, k, l, m, n):  # Take max
            with hcl.if_(V_new[i, j, k, l, m, n] < l0[i, j, k, l, m, n]):
                V_new[i, j, k, l, m, n] = l0[i, j, k, l, m, n]

        def minVWithV0(i, j, k, l, m, n):  # Take min
            with hcl.if_(V_new[i, j, k, l, m, n] > l0[i, j, k, l, m, n]):
                V_new[i, j, k, l, m, n] = l0[i, j, k, l, m, n]

        # Operations over time
        def minVWithVInit(i, j, k, l, m, n):
            with hcl.if_(V_new[i, j, k, l, m, n] > V_init[i, j, k, l, m, n]):
                V_new[i, j, k, l, m, n] = V_init[i, j, k, l, m, n]

        def maxVWithVInit(i, j, k, l, m, n):
            with hcl.if_(V_new[i, j, k, l, m, n] > V_init[i, j, k, l, m, n]):
                V_new[i, j, k, l, m, n] = V_init[i, j, k, l, m, n]

        # Calculate Hamiltonian for every grid point in V_init
        with hcl.Stage("Hamiltonian"):
            with hcl.for_(0, V_init.shape[0], name="i") as i:
                with hcl.for_(0, V_init.shape[1], name="j") as j:
                    with hcl.for_(0, V_init.shape[2], name="k") as k:
                        with hcl.for_(0, V_init.shape[3], name="l") as l:
                            with hcl.for_(0, V_init.shape[4], name="m") as m:
                                with hcl.for_(0, V_init.shape[5], name="n") as n:
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
                                    dV_dx5_L = hcl.scalar(0, "dV_dx5_L")
                                    dV_dx5_R = hcl.scalar(0, "dV_dx5_R")
                                    dV_dx5 = hcl.scalar(0, "dV_dx5")
                                    dV_dx6_L = hcl.scalar(0, "dV_dx6_L")
                                    dV_dx6_R = hcl.scalar(0, "dV_dx6_R")
                                    dV_dx6 = hcl.scalar(0, "dV_dx6")

                                    # No tensor slice operation
                                    # dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
                                    if accuracy == "low":
                                        dV_dx1_L[0], dV_dx1_R[0] = spa_derivX1_6d(i, j, k, l, m, n, V_init, g)
                                        dV_dx2_L[0], dV_dx2_R[0] = spa_derivX2_6d(i, j, k, l, m, n, V_init, g)
                                        dV_dx3_L[0], dV_dx3_R[0] = spa_derivX3_6d(i, j, k, l, m, n, V_init, g)
                                        dV_dx4_L[0], dV_dx4_R[0] = spa_derivX4_6d(i, j, k, l, m, n, V_init, g)
                                        dV_dx5_L[0], dV_dx5_R[0] = spa_derivX5_6d(i, j, k, l, m, n, V_init, g)
                                        dV_dx6_L[0], dV_dx6_R[0] = spa_derivX6_6d(i, j, k, l, m, n, V_init, g)
                                    if accuracy == "high":
                                        dV_dx1_L[0], dV_dx1_R[0] = secondOrderX1_6d(i, j, k, l, m, n, V_init, g)
                                        dV_dx2_L[0], dV_dx2_R[0] = secondOrderX2_6d(i, j, k, l, m, n, V_init, g)
                                        dV_dx3_L[0], dV_dx3_R[0] = secondOrderX3_6d(i, j, k, l, m, n, V_init, g)
                                        dV_dx4_L[0], dV_dx4_R[0] = secondOrderX4_6d(i, j, k, l, m, n, V_init, g)
                                        dV_dx5_L[0], dV_dx5_R[0] = secondOrderX5_6d(i, j, k, l, m, n, V_init, g)
                                        dV_dx6_L[0], dV_dx6_R[0] = secondOrderX6_6d(i, j, k, l, m, n, V_init, g)

                                    # Saves spatial derivative diff into tables
                                    deriv_diff1[i, j, k, l, m, n] = dV_dx1_R[0] - dV_dx1_L[0]
                                    deriv_diff2[i, j, k, l, m, n] = dV_dx2_R[0] - dV_dx2_L[0]
                                    deriv_diff3[i, j, k, l, m, n] = dV_dx3_R[0] - dV_dx3_L[0]
                                    deriv_diff4[i, j, k, l, m, n] = dV_dx4_R[0] - dV_dx4_L[0]
                                    deriv_diff5[i, j, k, l, m, n] = dV_dx5_R[0] - dV_dx5_L[0]
                                    deriv_diff6[i, j, k, l, m, n] = dV_dx6_R[0] - dV_dx6_L[0]

                                    # Calculate average gradient
                                    dV_dx1[0] = (dV_dx1_L + dV_dx1_R) / 2
                                    dV_dx2[0] = (dV_dx2_L + dV_dx2_R) / 2
                                    dV_dx3[0] = (dV_dx3_L + dV_dx3_R) / 2
                                    dV_dx4[0] = (dV_dx4_L + dV_dx4_R) / 2
                                    dV_dx5[0] = (dV_dx5_L + dV_dx5_R) / 2
                                    dV_dx6[0] = (dV_dx6_L + dV_dx6_R) / 2

                                    # Find optimal control
                                    uOpt = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), (
                                    dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0], dV_dx5[0], dV_dx6[0]))
                                    # Find optimal disturbance
                                    dOpt = my_object.opt_dstb(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), (
                                    dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0], dV_dx5[0], dV_dx6[0]))

                                    # Find rates of changes based on dynamics equation
                                    dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt, dx6_dt = my_object.dynamics(t, (
                                    x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), uOpt, dOpt)

                                    # Calculate Hamiltonian terms:
                                    V_new[i, j, k, l, m, n] = -(
                                                dx1_dt * dV_dx1[0] + dx2_dt * dV_dx2[0] + dx3_dt * dV_dx3[0] + dx4_dt *
                                                dV_dx4[0] + dx5_dt * dV_dx5[0] + dx6_dt * dV_dx6[0])

                                    # Get derivMin
                                    with hcl.if_(dV_dx1_L[0] < min_deriv1[0]):
                                        min_deriv1[0] = dV_dx1_L[0]
                                    with hcl.if_(dV_dx1_R[0] < min_deriv1[0]):
                                        min_deriv1[0] = dV_dx1_R[0]

                                    with hcl.if_(dV_dx2_L[0] < min_deriv2[0]):
                                        min_deriv2[0] = dV_dx2_L[0]
                                    with hcl.if_(dV_dx2_R[0] < min_deriv2[0]):
                                        min_deriv2[0] = dV_dx2_R[0]

                                    with hcl.if_(dV_dx3_L[0] < min_deriv3[0]):
                                        min_deriv3[0] = dV_dx3_L[0]
                                    with hcl.if_(dV_dx3_R[0] < min_deriv3[0]):
                                        min_deriv3[0] = dV_dx3_R[0]

                                    with hcl.if_(dV_dx4_L[0] < min_deriv4[0]):
                                        min_deriv4[0] = dV_dx4_L[0]
                                    with hcl.if_(dV_dx4_R[0] < min_deriv4[0]):
                                        min_deriv4[0] = dV_dx4_R[0]

                                    with hcl.if_(dV_dx5_L[0] < min_deriv5[0]):
                                        min_deriv5[0] = dV_dx5_L[0]
                                    with hcl.if_(dV_dx5_R[0] < min_deriv5[0]):
                                        min_deriv5[0] = dV_dx5_R[0]

                                    with hcl.if_(dV_dx6_L[0] < min_deriv6[0]):
                                        min_deriv6[0] = dV_dx6_L[0]
                                    with hcl.if_(dV_dx6_R[0] < min_deriv6[0]):
                                        min_deriv6[0] = dV_dx6_R[0]

                                    # Get derivMax
                                    with hcl.if_(dV_dx1_L[0] > max_deriv1[0]):
                                        max_deriv1[0] = dV_dx1_L[0]
                                    with hcl.if_(dV_dx1_R[0] > max_deriv1[0]):
                                        max_deriv1[0] = dV_dx1_R[0]

                                    with hcl.if_(dV_dx2_L[0] > max_deriv2[0]):
                                        max_deriv2[0] = dV_dx2_L[0]
                                    with hcl.if_(dV_dx2_R[0] > max_deriv2[0]):
                                        max_deriv2[0] = dV_dx2_R[0]

                                    with hcl.if_(dV_dx3_L[0] > max_deriv3[0]):
                                        max_deriv3[0] = dV_dx3_L[0]
                                    with hcl.if_(dV_dx3_R[0] > max_deriv3[0]):
                                        max_deriv3[0] = dV_dx3_R[0]

                                    with hcl.if_(dV_dx4_L[0] > max_deriv4[0]):
                                        max_deriv4[0] = dV_dx4_L[0]
                                    with hcl.if_(dV_dx4_R[0] > max_deriv4[0]):
                                        max_deriv4[0] = dV_dx4_R[0]

                                    with hcl.if_(dV_dx5_L[0] > max_deriv5[0]):
                                        max_deriv5[0] = dV_dx5_L[0]
                                    with hcl.if_(dV_dx5_R[0] > max_deriv5[0]):
                                        max_deriv5[0] = dV_dx5_R[0]

                                    with hcl.if_(dV_dx6_L[0] > max_deriv6[0]):
                                        max_deriv6[0] = dV_dx6_L[0]
                                    with hcl.if_(dV_dx6_R[0] > max_deriv6[0]):
                                        max_deriv6[0] = dV_dx6_R[0]

        # Calculate dissipation amount
        with hcl.Stage("Dissipation"):
            uOptL1 = hcl.scalar(0, "uOptL1")
            uOptL2 = hcl.scalar(0, "uOptL2")
            uOptL3 = hcl.scalar(0, "uOptL3")
            uOptL4 = hcl.scalar(0, "uOptL4")

            uOptU1 = hcl.scalar(0, "uOptU1")
            uOptU2 = hcl.scalar(0, "uOptU2")
            uOptU3 = hcl.scalar(0, "uOptU3")
            uOptU4 = hcl.scalar(0, "uOptU4")

            dOptL1 = hcl.scalar(0, "dOptL1")
            dOptL2 = hcl.scalar(0, "dOptL2")
            dOptL3 = hcl.scalar(0, "dOptL3")
            dOptL4 = hcl.scalar(0, "dOptL4")

            dOptU1 = hcl.scalar(0, "dOptU1")
            dOptU2 = hcl.scalar(0, "dOptU2")
            dOptU3 = hcl.scalar(0, "dOptU3")
            dOptU4 = hcl.scalar(0, "dOptU4")

            # Storing alphas
            alpha1 = hcl.scalar(0, "alpha1")
            alpha2 = hcl.scalar(0, "alpha2")
            alpha3 = hcl.scalar(0, "alpha3")
            alpha4 = hcl.scalar(0, "alpha4")
            alpha5 = hcl.scalar(0, "alpha5")
            alpha6 = hcl.scalar(0, "alpha6")

            """ 
                NOTE: If optimal adversarial disturbance is not dependent on states
                , the below approximate LOWER/UPPER BOUND optimal disturbance is  accurate.
                If that's not the case, move the next two statements into the nested loops and modify the states passed in 
                as my_object.opt_dstb(t, (x1[i], x2[j], x3[k], x4[l], ...), ...).
                The reason we don't have this line in the nested loop by default is to avoid redundant computations
                for certain systems where disturbance are not dependent on states.
                In general, dissipation amount can just be approximates.  
            """

            # Find LOWER BOUND optimal disturbance
            dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0] = my_object.opt_dstb(t, (x1[0], x2[0], x3[0], x4[0], x5[0], x6[0]),
                (min_deriv1[0], min_deriv2[0], min_deriv3[0], min_deriv4[0], min_deriv5[0], min_deriv6[0]))
            # Find UPPER BOUND optimal disturbance
            dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0] = my_object.opt_dstb(t, (x1[0], x2[0], x3[0], x4[0], x5[0], x6[0]),
                (max_deriv1[0], max_deriv2[0], max_deriv3[0], max_deriv4[0], max_deriv5[0], max_deriv6[0]))
            with hcl.for_(0, V_init.shape[0], name="i") as i:
                with hcl.for_(0, V_init.shape[1], name="j") as j:
                    with hcl.for_(0, V_init.shape[2], name="k") as k:
                        with hcl.for_(0, V_init.shape[3], name="l") as l:
                            with hcl.for_(0, V_init.shape[4], name="m") as m:
                                with hcl.for_(0, V_init.shape[5], name="n") as n:
                                    dx_LL1 = hcl.scalar(0, "dx_LL1")
                                    dx_LL2 = hcl.scalar(0, "dx_LL2")
                                    dx_LL3 = hcl.scalar(0, "dx_LL3")
                                    dx_LL4 = hcl.scalar(0, "dx_LL4")
                                    dx_LL5 = hcl.scalar(0, "dx_LL5")
                                    dx_LL6 = hcl.scalar(0, "dx_LL6")

                                    dx_UL1 = hcl.scalar(0, "dx_UL1")
                                    dx_UL2 = hcl.scalar(0, "dx_UL2")
                                    dx_UL3 = hcl.scalar(0, "dx_UL3")
                                    dx_UL4 = hcl.scalar(0, "dx_UL4")
                                    dx_UL5 = hcl.scalar(0, "dx_UL5")
                                    dx_UL6 = hcl.scalar(0, "dx_UL6")
                                    #
                                    dx_LU1 = hcl.scalar(0, "dx_LU1")
                                    dx_LU2 = hcl.scalar(0, "dx_LU2")
                                    dx_LU3 = hcl.scalar(0, "dx_LU3")
                                    dx_LU4 = hcl.scalar(0, "dx_LU4")
                                    dx_LU5 = hcl.scalar(0, "dx_LU5")
                                    dx_LU6 = hcl.scalar(0, "dx_LU6")

                                    dx_UU1 = hcl.scalar(0, "dx_UU1")
                                    dx_UU2 = hcl.scalar(0, "dx_UU2")
                                    dx_UU3 = hcl.scalar(0, "dx_UU3")
                                    dx_UU4 = hcl.scalar(0, "dx_UU4")
                                    dx_UU5 = hcl.scalar(0, "dx_UU5")
                                    dx_UU6 = hcl.scalar(0, "dx_UU6")

                                    # Find LOWER BOUND optimal control
                                    uOptL1[0], uOptL2[0], uOptL3[0], uOptL4[0] = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), (
                                    min_deriv1[0], min_deriv2[0], min_deriv3[0], min_deriv4[0], min_deriv5[0],
                                    min_deriv6[0]))
                                    # Find UPPER BOUND optimal control
                                    uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0] = my_object.opt_ctrl(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), (
                                    max_deriv1[0], max_deriv2[0], max_deriv3[0], max_deriv4[0], max_deriv5[0],
                                    max_deriv6[0]))

                                    # Get upper bound and lower bound rates of changes
                                    dx_LL1[0], dx_LL2[0], dx_LL3[0], dx_LL4[0], dx_LL5[0], dx_LL6[
                                        0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), (uOptL1[0], uOptL2[0], uOptL3[0], uOptL4[0]),
                                                                (dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0]))
                                    # Get absolute value of each
                                    dx_LL1[0] = my_abs(dx_LL1[0])
                                    dx_LL2[0] = my_abs(dx_LL2[0])
                                    dx_LL3[0] = my_abs(dx_LL3[0])
                                    dx_LL4[0] = my_abs(dx_LL4[0])
                                    dx_LL5[0] = my_abs(dx_LL5[0])
                                    dx_LL6[0] = my_abs(dx_LL6[0])

                                    dx_UL1[0], dx_UL2[0], dx_UL3[0], dx_UL4[0], dx_UL5[0], dx_UL6[
                                        0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), (uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0]),
                                                                (dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0]))
                                    # Get absolute value of each
                                    dx_UL1[0] = my_abs(dx_UL1[0])
                                    dx_UL2[0] = my_abs(dx_UL2[0])
                                    dx_UL3[0] = my_abs(dx_UL3[0])
                                    dx_UL4[0] = my_abs(dx_UL4[0])
                                    dx_UL5[0] = my_abs(dx_UL5[0])
                                    dx_UL6[0] = my_abs(dx_UL6[0])

                                    # Set maximum alphas
                                    alpha1[0] = my_max(dx_UL1[0], dx_LL1[0])
                                    alpha2[0] = my_max(dx_UL2[0], dx_LL2[0])
                                    alpha3[0] = my_max(dx_UL3[0], dx_LL3[0])
                                    alpha4[0] = my_max(dx_UL4[0], dx_LL4[0])
                                    alpha5[0] = my_max(dx_UL5[0], dx_LL5[0])
                                    alpha6[0] = my_max(dx_UL6[0], dx_LL6[0])

                                    dx_LU1[0], dx_LU2[0], dx_LU3[0], dx_LU4[0], dx_LU5[0], dx_LU6[
                                        0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), (uOptL1[0], uOptL2[0], uOptL3[0], uOptL4[0]),
                                                                (dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0]))
                                    # Get absolute value of each
                                    dx_LU1[0] = my_abs(dx_LU1[0])
                                    dx_LU2[0] = my_abs(dx_LU2[0])
                                    dx_LU3[0] = my_abs(dx_LU3[0])
                                    dx_LU4[0] = my_abs(dx_LU4[0])
                                    dx_LU5[0] = my_abs(dx_LU5[0])
                                    dx_LU6[0] = my_abs(dx_LU6[0])

                                    alpha1[0] = my_max(alpha1[0], dx_LU1[0])
                                    alpha2[0] = my_max(alpha2[0], dx_LU2[0])
                                    alpha3[0] = my_max(alpha3[0], dx_LU3[0])
                                    alpha4[0] = my_max(alpha4[0], dx_LU4[0])
                                    alpha5[0] = my_max(alpha5[0], dx_LU5[0])
                                    alpha6[0] = my_max(alpha6[0], dx_LU6[0])

                                    dx_UU1[0], dx_UU2[0], dx_UU3[0], dx_UU4[0], dx_UU5[0], dx_UU6[
                                        0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), (uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0]),
                                                                (dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0]))
                                    dx_UU1[0] = my_abs(dx_UU1[0])
                                    dx_UU2[0] = my_abs(dx_UU2[0])
                                    dx_UU3[0] = my_abs(dx_UU3[0])
                                    dx_UU4[0] = my_abs(dx_UU4[0])
                                    dx_UU5[0] = my_abs(dx_UU5[0])
                                    dx_UU6[0] = my_abs(dx_UU6[0])

                                    alpha1[0] = my_max(alpha1[0], dx_UU1[0])
                                    alpha2[0] = my_max(alpha2[0], dx_UU2[0])
                                    alpha3[0] = my_max(alpha3[0], dx_UU3[0])
                                    alpha4[0] = my_max(alpha4[0], dx_UU4[0])
                                    alpha5[0] = my_max(alpha5[0], dx_UU5[0])
                                    alpha6[0] = my_max(alpha6[0], dx_UU6[0])

                                    diss = hcl.scalar(0, "diss")
                                    diss[0] = 0.5 * (deriv_diff1[i, j, k, l, m, n] * alpha1[0] + deriv_diff2[
                                        i, j, k, l, m, n] * alpha2[0] \
                                                     + deriv_diff3[i, j, k, l, m, n] * alpha3[0] + deriv_diff4[
                                                         i, j, k, l, m, n] * alpha4[0] \
                                                     + deriv_diff5[i, j, k, l, m, n] * alpha5[0] + deriv_diff6[
                                                         i, j, k, l, m, n] * alpha6[0])

                                    # Finally
                                    V_new[i, j, k, l, m, n] = -(V_new[i, j, k, l, m, n] - diss[0])
                                    # Get maximum alphas in each dimension

                                    # Calculate alphas
                                    with hcl.if_(alpha1 > max_alpha1):
                                        max_alpha1[0] = alpha1[0]
                                    with hcl.if_(alpha2 > max_alpha2):
                                        max_alpha2[0] = alpha2[0]
                                    with hcl.if_(alpha3 > max_alpha3):
                                        max_alpha3[0] = alpha3[0]
                                    with hcl.if_(alpha4 > max_alpha4):
                                        max_alpha4[0] = alpha4[0]
                                    with hcl.if_(alpha5 > max_alpha5):
                                        max_alpha5[0] = alpha5[0]
                                    with hcl.if_(alpha6 > max_alpha6):
                                        max_alpha6[0] = alpha6[0]

        # Determine time step
        delta_t = hcl.compute((1,), lambda x: step_bound(), name="delta_t")
        # hcl.update(t, lambda x: t[x] + delta_t[x])

        # Integrate
        # if compMethod == 'HJ_PDE':
        result = hcl.update(V_new,
                            lambda i, j, k, l, m, n: V_init[i, j, k, l, m, n] + V_new[i, j, k, l, m, n] * delta_t[0])
        if compMethod == 'maxVWithV0' or compMethod == 'maxVWithVTarget':
            result = hcl.update(V_new, lambda i, j, k, l, m, n: maxVWithV0(i, j, k, l, m, n))
        if compMethod == 'minVWithV0' or compMethod == 'minVWithVTarget':
            result = hcl.update(V_new, lambda i, j, k, l, m, n: minVWithV0(i, j, k, l, m, n))
        if compMethod == 'maxVWithVInit':
            result = hcl.update(V_new, lambda i, j, k, l, m, n: maxVWithVInit(i, j, k, l, m, n))
        if compMethod == 'minVWithVInit':
            result = hcl.update(V_new, lambda i, j, k, l, m, n: minVWithVInit(i, j, k, l, m, n))
        # Copy V_new to V_init
        hcl.update(V_init, lambda i, j, k, l, m, n: V_new[i, j, k, l, m, n])
        return result


    s = hcl.create_schedule([V_f, V_init, x1, x2, x3, x4, x5, x6, t, l0], graph_create)
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

    # Return executable
    return (hcl.build(s))
