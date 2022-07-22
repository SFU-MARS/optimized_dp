import heterocl as hcl
import numpy as np
import time
import plotly.graph_objects as go
from GridProcessing import Grid
from ShapesFunctions import *
from CustomGraphFunctions import *
from ROV_WaveBwds_6D import *
from argparse import ArgumentParser

import scipy.io as sio


""" USER INTERFACES
- Define grid

- Generate initial values for grid using shape functions

- Time length for computations

- Run
"""

# Grid field in this order: x_a, z_a, u_r, w_r, x, z

g = grid(np.array([-5.0, -5.0, -0.5, -0.7, -5.0, 2.0]), np.array([5.0, 5.0, 0.5, 0.7, 5.0, 12.0]), 6, np.array([25,25,25, 25, 25, 25])) # Leave out periodic field
# Use the grid to initialize initial value function
my_shape = Cylinder6D(g, [3, 4, 5, 6])
# Define my object
myROV_6D = ROV_WaveBwds_6D()

# Look-back lenght and time step
lookback_length = 20.00
t_step = 0.2

tau = np.arange(start = 0, stop = lookback_length + t_step, step = t_step)
print("I'm here \n")



# Note that t has 2 elements t1, t2
def graph_6D(V_new, V_init, deriv_diff1, deriv_diff2, deriv_diff3, deriv_diff4, deriv_diff5, deriv_diff6,
             x1, x2, x3, x4, x5, x6 ,t):
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
    
    def step_bound(): # Function to calculate time step
        stepBoundInv = hcl.scalar(0, "stepBoundInv")
        stepBound    = hcl.scalar(0, "stepBound")
        stepBoundInv[0] = max_alpha1[0]/g.dx[0] + max_alpha2[0]/g.dx[1] + max_alpha3[0]/g.dx[2] + max_alpha4[0]/g.dx[3] \
                            + max_alpha5[0]/g.dx[4] + max_alpha6[0]/g.dx[5]

        stepBound[0] = 0.8/stepBoundInv[0]
        with hcl.if_(stepBound > t[1] - t[0]):
            stepBound[0] = t[1] - t[0]

        # Update the lower time ranges
        t[0] = t[0] + stepBound[0]
        return stepBound[0]

    # Calculate Hamiltonian for every grid point in V_init
    with hcl.Stage("Hamiltonian"):
        with hcl.for_(0, V_init.shape[0], name="i") as i:
            with hcl.for_(0, V_init.shape[1], name="j") as j:
                with hcl.for_(0, V_init.shape[2], name="k") as k:
                    with hcl.for_(0, V_init.shape[3], name="l") as l:
                        with hcl.for_(0, V_init.shape[4], name="m") as m:
                            with hcl.for_(0, V_init.shape[5], name="n") as n:
                                #Variables to calculate dV_dx
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
                                #dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
                                dV_dx1_L[0], dV_dx1_R[0] = spa_derivX6_6d(i, j, k, l, m, n, V_init, g)
                                dV_dx2_L[0], dV_dx2_R[0] = spa_derivX5_6d(i, j, k, l, m, n, V_init, g)
                                dV_dx3_L[0], dV_dx3_R[0] = spa_derivX4_6d(i, j, k, l, m, n, V_init, g)
                                dV_dx4_L[0], dV_dx4_R[0] = spa_derivX3_6d(i, j, k, l, m, n, V_init, g)
                                dV_dx5_L[0], dV_dx5_R[0] = spa_derivX2_6d(i, j, k, l, m, n, V_init, g)
                                dV_dx6_L[0], dV_dx6_R[0] = spa_derivX1_6d(i, j, k, l, m, n, V_init, g)

                                # Saves spatial derivative diff into tables
                                deriv_diff1[i, j, k, l, m, n] = dV_dx1_R[0] - dV_dx1_L[0]
                                deriv_diff2[i, j, k, l, m, n] = dV_dx2_R[0] - dV_dx2_L[0]
                                deriv_diff3[i, j, k, l, m, n] = dV_dx3_R[0] - dV_dx3_L[0]
                                deriv_diff4[i, j, k, l, m, n] = dV_dx4_R[0] - dV_dx4_L[0]
                                deriv_diff5[i, j, k, l, m, n] = dV_dx5_R[0] - dV_dx5_L[0]
                                deriv_diff6[i, j, k, l, m, n] = dV_dx6_R[0] - dV_dx6_L[0]

                                #Calculate average gradient
                                dV_dx1[0] = (dV_dx1_L + dV_dx1_R) / 2
                                dV_dx2[0] = (dV_dx2_L + dV_dx2_R) / 2
                                dV_dx3[0] = (dV_dx3_L + dV_dx3_R) / 2
                                dV_dx4[0] = (dV_dx4_L + dV_dx4_R) / 2
                                dV_dx5[0] = (dV_dx5_L + dV_dx5_R) / 2
                                dV_dx6[0] = (dV_dx6_L + dV_dx6_R) / 2


                                # Find optimal control
                                uOpt = myROV_6D.opt_ctrl(t,(dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0], dV_dx5[0], dV_dx6[0]))
                                # Find optimal disturbance
                                dOpt = myROV_6D.opt_dstb((dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0], dV_dx5[0], dV_dx6[0]))

                                # Find rates of changes based on dynamics equation
                                dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt, dx6_dt = myROV_6D.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), uOpt, dOpt)

                                # # Calculate Hamiltonian terms:
                                V_new[i, j, k, l, m, n] =  - myROV_6D.Hamiltonian((dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt, dx6_dt), (dV_dx1, dV_dx2, dV_dx3, dV_dx4, dV_dx5, dV_dx6))

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


        # Find LOWER BOUND optimal control
        uOptL1[0], uOptL2[0], uOptL3[0], uOptL4[0] = myROV_6D.opt_ctrl(t, (min_deriv1[0], min_deriv2[0], min_deriv3[0], min_deriv4[0], min_deriv5[0], min_deriv6[0]))
        # Find UPPER BOUND optimal control
        uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0] = myROV_6D.opt_ctrl(t, (max_deriv1[0], max_deriv2[0], max_deriv3[0], max_deriv4[0], max_deriv5[0], max_deriv6[0]))
        # Find LOWER BOUND optimal disturbance
        dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0] = myROV_6D.opt_dstb((min_deriv1[0], min_deriv2[0], min_deriv3[0], min_deriv4[0], min_deriv5[0], min_deriv6[0]))
        # Find UPPER BOUND optimal disturbance
        dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0] = myROV_6D.opt_dstb((max_deriv1[0], max_deriv2[0], max_deriv3[0], max_deriv4[0], min_deriv5[0], min_deriv6[0]))

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

                                # Get upper bound and lower bound rates of changes
                                dx_LL1[0], dx_LL2[0], dx_LL3[0], dx_LL4[0], dx_LL5[0], dx_LL6[0] = myROV_6D.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]) \
                                                                                                                     , (uOptL1[0], uOptL2[0], uOptL3[0], uOptL4[0]), (dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0]))
                                # Get absolute value of each
                                dx_LL1[0] = my_abs(dx_LL1[0])
                                dx_LL2[0] = my_abs(dx_LL2[0])
                                dx_LL3[0] = my_abs(dx_LL3[0])
                                dx_LL4[0] = my_abs(dx_LL4[0])
                                dx_LL5[0] = my_abs(dx_LL5[0])
                                dx_LL6[0] = my_abs(dx_LL6[0])

                                dx_UL1[0], dx_UL2[0], dx_UL3[0], dx_UL4[0], dx_UL5[0], dx_UL6[0] = myROV_6D.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n])\
                                                                                                                     , (uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0]), (dOptL1[0], dOptL2[0], dOptL3[0], dOptL4[0]))
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

                                dx_LU1[0], dx_LU2[0], dx_LU3[0], dx_LU4[0], dx_LU5[0], dx_LU6[0] = myROV_6D.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n])\
                                                                                                                     , (uOptL1[0], uOptL2[0], uOptL3[0], uOptL4[0]), (dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0]))
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

                                dx_UU1[0], dx_UU2[0], dx_UU3[0], dx_UU4[0], dx_UU5[0], dx_UU6[0] = myROV_6D.dynamics(t, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n])\
                                                                                                                     , (uOptU1[0], uOptU2[0], uOptU3[0], uOptU4[0]), (dOptU1[0], dOptU2[0], dOptU3[0], dOptU4[0]))
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
                                diss[0] = 0.5*(deriv_diff1[i, j, k, l, m, n]*alpha1[0] + deriv_diff2[i, j, k, l, m, n]*alpha2[0] \
                                               + deriv_diff3[i, j, k, l, m, n]* alpha3[0] + deriv_diff4[i, j, k, l, m, n]* alpha4[0] \
                                               + deriv_diff5[i, j, k, l, m, n]* alpha5[0] + deriv_diff6[i, j, k, l, m, n]* alpha6[0])

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
    #hcl.update(t, lambda x: t[x] + delta_t[x])
    # Integrate
    result = hcl.update(V_new, lambda i, j, k, l, m, n: V_init[i, j, k, l, m, n] + V_new[i, j, k, l, m, n] * delta_t[0])
    # Copy V_new to V_init
    hcl.update(V_init, lambda i, j, k, l, m, n: V_new[i, j, k, l, m, n])
    return result

def main():
    ################### PARSING ARGUMENTS FROM USERS #####################

    parser = ArgumentParser()
    parser.add_argument("-p", "--plot", default=False, type=bool)
    # Print out LLVM option only
    parser.add_argument("-l", "--llvm", default=False, type=bool)
    args = parser.parse_args()

    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    ################## DATA SHAPE PREPARATION FOR GRAPH FORMATION  ####################
    V_f = hcl.placeholder(tuple(g.pts_each_dim), name="V_f", dtype = hcl.Float())
    V_init = hcl.placeholder(tuple(g.pts_each_dim), name="V_init", dtype=hcl.Float())
    #x = hcl.placeholder((6, g.pts_each_dim[0]), name="x", dtype=hcl.Float())
    t = hcl.placeholder((2,), name="t", dtype=hcl.Float())

    # Deriv diff tensor
    deriv_diff1 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff1")
    deriv_diff2 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff2")
    deriv_diff3 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff3")
    deriv_diff4 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff4")
    deriv_diff5 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff5")
    deriv_diff6 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff6")

    # Positions vector
    x1 = hcl.placeholder((g.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((g.pts_each_dim[1],), name="x2", dtype=hcl.Float())
    x3 = hcl.placeholder((g.pts_each_dim[2],), name="x3", dtype=hcl.Float())
    x4 = hcl.placeholder((g.pts_each_dim[3],), name="x4", dtype=hcl.Float())
    x5 = hcl.placeholder((g.pts_each_dim[4],), name="x5", dtype=hcl.Float())
    x6 = hcl.placeholder((g.pts_each_dim[5],), name="x6", dtype=hcl.Float())


    ##################### CREATE SCHEDULE##############

    # Create schedule
    s = hcl.create_schedule(
        [V_f, V_init, deriv_diff1, deriv_diff2, deriv_diff3, deriv_diff4, deriv_diff5, deriv_diff6, x1, x2, x3, x4, x5, x6, t], graph_6D)

    # Inspect the LLVM code
    print(hcl.lower(s))



    ################# INITIALIZE DATA TO BE INPUT INTO GRAPH ##########################

    print("Initializing\n")

    V_0 = hcl.asarray(my_shape)
    V_1=  hcl.asarray(np.zeros(tuple(g.pts_each_dim)))

    list_x1 = np.reshape(g.vs[0], g.pts_each_dim[0])
    list_x2 = np.reshape(g.vs[1], g.pts_each_dim[1])
    list_x3 = np.reshape(g.vs[2], g.pts_each_dim[2])
    list_x4 = np.reshape(g.vs[3], g.pts_each_dim[3])
    list_x5 = np.reshape(g.vs[4], g.pts_each_dim[4])
    list_x6 = np.reshape(g.vs[5], g.pts_each_dim[5])

    # Convert to hcl array type
    list_x1 = hcl.asarray(list_x1)
    list_x2 = hcl.asarray(list_x2)
    list_x3 = hcl.asarray(list_x3)
    list_x4 = hcl.asarray(list_x4)
    list_x5 = hcl.asarray(list_x5)
    list_x6 = hcl.asarray(list_x6)

    # Initialize deriv diff tensor
    deriv_diff1 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    deriv_diff2 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    deriv_diff3 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    deriv_diff4 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    deriv_diff5 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    deriv_diff6 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))


    ##################### CODE OPTIMIZATION HERE ###########################
    print("Optimizing\n")

    # Accessing the hamiltonian stage
    s_H = graph_6D.Hamiltonian
    s_D = graph_6D.Dissipation

    #
    s[s_H].parallel(s_H.i)
    s[s_D].parallel(s_D.i)

    # Inspect IR
    #if args.llvm:
    #    print(hcl.lower(s))

    ################ GET EXECUTABLE AND USE THE EXECUTABLE ############
    print("Running\n")

    # Get executable
    solve_pde = hcl.build(s)

    # Variables used for timing
    execution_time = 0
    lookback_time = 0

    tNow = tau[0]
    for i in range (1, len(tau)):
        #tNow = tau[i-1]
        t_minh= hcl.asarray(np.array((tNow, tau[i])))
        while tNow <= tau[i] - 1e-4:
             # Start timing
             start = time.time()

             # Run the execution and pass input into graph
             solve_pde(V_1, V_0, deriv_diff1, deriv_diff2, deriv_diff3, deriv_diff4, deriv_diff5, deriv_diff6,
                  list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, t_minh)

             tNow = np.asscalar((t_minh.asnumpy())[0])

             if lookback_time != 0: # Exclude first time of the computation
                 execution_time += time.time() - start

             # Some information printing
             print(t_minh)
             print("Computational time to integrate (s): {:.5f}".format(time.time() - start))

             # Saving data into disk
             if tNow >= tau[i] - 1e-4:
                print("Saving files\n")
                sio.savemat('/local-scratch/ROV_6d_TEB/V_value_{:d}.mat'.format(i), {'V_array': V_1.asnumpy()})

    
    #print(V_1.asnumpy())
    #
    # V_1 = V_1.asnumpy()
    # # V_1 = np.swapaxes(V_1, 0,2)
    # #V = np.swapaxes(V, 1,2)
    # #probe = probe.asnumpy()
    # #probe = np.swapaxes(probe, 0, 2)
    # #probe = np.swapaxes(probe, 1, 2)
    # #print(V)
    # #V_1 = V_1.asnumpy()
    #
    #
    # Time info printing
    print("Total kernel time (s): {:.5f}".format(execution_time))
    print("Finished solving\n")

    ##################### PLOTTING #####################
    if args.plot:
        print("Plotting beautiful plots. Please wait\n")
        fig = go.Figure(data=go.Isosurface(
            x=g.mg_X.flatten(),
            y=g.mg_Y.flatten(),
            z=g.mg_T.flatten(),
            value=V_1.flatten(),
            colorscale='jet',
            isomin=0,
            surface_count=1,
            isomax=0,
            caps=dict(x_show=True, y_show=True)
        ))
        fig.show()
        print("Please check the plot on your browser.")


if __name__ == '__main__':
  main()
