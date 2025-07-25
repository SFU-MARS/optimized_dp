import heterocl as hcl
import numpy as np
from odp.computeGraphs.graph_6D import *

######################################### HELPER FUNCTIONS #########################################

# Update the phi function at position (i,j,k,l,m,n)
def updatePhi(i, j, k, l, m, n, my_object, phi, constraint, g, x1, x2, x3, x4, x5, x6):
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

    sigma1 = hcl.scalar(0, "sigma1")
    sigma2 = hcl.scalar(0, "sigma2")
    sigma3 = hcl.scalar(0, "sigma3")
    sigma4 = hcl.scalar(0, "sigma4")
    sigma5 = hcl.scalar(0, "sigma5")
    sigma6 = hcl.scalar(0, "sigma6")

    # dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
    dV_dx1_L[0], dV_dx1_R[0] = spa_derivX1_6d(i, j, k, l, m, n, phi, g)
    dV_dx2_L[0], dV_dx2_R[0] = spa_derivX2_6d(i, j, k, l, m, n, phi, g)
    dV_dx3_L[0], dV_dx3_R[0] = spa_derivX3_6d(i, j, k, l, m, n, phi, g)
    dV_dx4_L[0], dV_dx4_R[0] = spa_derivX4_6d(i, j, k, l, m, n, phi, g)
    dV_dx5_L[0], dV_dx5_R[0] = spa_derivX5_6d(i, j, k, l, m, n, phi, g)
    dV_dx6_L[0], dV_dx6_R[0] = spa_derivX6_6d(i, j, k, l, m, n, phi, g)

    # Calculate average gradient
    dV_dx1[0] = (dV_dx1_L[0] + dV_dx1_R[0]) / 2
    dV_dx2[0] = (dV_dx2_L[0] + dV_dx2_R[0]) / 2
    dV_dx3[0] = (dV_dx3_L[0] + dV_dx3_R[0]) / 2
    dV_dx4[0] = (dV_dx4_L[0] + dV_dx4_R[0]) / 2
    dV_dx5[0] = (dV_dx5_L[0] + dV_dx5_R[0]) / 2
    dV_dx6[0] = (dV_dx6_L[0] + dV_dx6_R[0]) / 2

    # Find the optimal control through my_object's API
    uOpt = my_object.opt_ctrl(0, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]),
                              (dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0], dV_dx5[0], dV_dx6[0]))
    # Find optimal disturbance
    dOpt = my_object.opt_dstb(0, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]),
                              (dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0], dV_dx5[0], dV_dx6[0]))

    # Find rates of changes based on dynamics equation
    dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt, dx6_dt = my_object.dynamics(0, (x1[i], x2[j], x3[k], x4[l], x5[m], x6[n]), uOpt, dOpt)

    H = hcl.scalar(0, "H")
    phiNew = hcl.scalar(0, "phiNew")
    diss1 = hcl.scalar(0, "diss1")
    diss2 = hcl.scalar(0, "diss2")
    diss3 = hcl.scalar(0, "diss3")
    diss4 = hcl.scalar(0, "diss4")
    diss5 = hcl.scalar(0, "diss5")
    diss6 = hcl.scalar(0, "diss6")

    # Calculate Hamiltonian terms:
    H[0] = (-(dx1_dt * dV_dx1[0] + dx2_dt * dV_dx2[0] + dx3_dt * dV_dx3[0] + dx4_dt * dV_dx4[0]
              + dx5_dt * dV_dx5[0] + dx6_dt * dV_dx6[0] + 1))

    # Calculate the "dissipation"
    sigma1[0] = my_abs(dx1_dt)
    sigma2[0] = my_abs(dx2_dt)
    sigma3[0] = my_abs(dx3_dt)
    sigma4[0] = my_abs(dx4_dt)
    sigma5[0] = my_abs(dx5_dt)
    sigma6[0] = my_abs(dx6_dt)

    c = hcl.scalar(0, "c")
    c[0] = sigma1[0] / g.dx[0] + sigma2[0] / g.dx[1] + sigma3[0] / g.dx[2] \
           + sigma4[0] / g.dx[3] + sigma5[0] / g.dx[4] + sigma6[0] / g.dx[5]

    diss1[0] = sigma1[0] * ((dV_dx1_R[0] - dV_dx1_L[0]) / 2 + phi[i, j, k, l, m, n] / g.dx[0])
    diss2[0] = sigma2[0] * ((dV_dx2_R[0] - dV_dx2_L[0]) / 2 + phi[i, j, k, l, m, n] / g.dx[1])
    diss3[0] = sigma3[0] * ((dV_dx3_R[0] - dV_dx3_L[0]) / 2 + phi[i, j, k, l, m, n] / g.dx[2])
    diss4[0] = sigma4[0] * ((dV_dx4_R[0] - dV_dx4_L[0]) / 2 + phi[i, j, k, l, m, n] / g.dx[3])
    diss5[0] = sigma5[0] * ((dV_dx5_R[0] - dV_dx5_L[0]) / 2 + phi[i, j, k, l, m, n] / g.dx[4])
    diss6[0] = sigma6[0] * ((dV_dx6_R[0] - dV_dx6_L[0]) / 2 + phi[i, j, k, l, m, n] / g.dx[5])

    # New phi
    phiNew[0] = (-H[0] + diss1[0] + diss2[0] + diss3[0] + diss4[0] + diss5[0] + diss6[0]) / c[0]
    #debugger[i, j, k, l, m, n] = phiNew[0]
    phi[i, j, k, l, m, n] = my_min(phi[i, j, k, l, m, n], phiNew[0])
    # Check with the obstacle
    phi[i, j, k, l, m, n] = my_max(phi[i, j, k, l, m, n], constraint[i, j, k, l, m, n])


def EvalBoundary(phi, constraint, g): 
    if 0 not in g.pDim:
        with hcl.for_(0, phi.shape[1], name="j") as j:
            with hcl.for_(0, phi.shape[2], name="k") as k:
                with hcl.for_(0, phi.shape[3], name="l") as l:
                    with hcl.for_(0, phi.shape[4], name="m") as m:
                        with hcl.for_(0, phi.shape[5], name="n") as n:
                            tmp1 = hcl.scalar(0, "tmp1")
                            tmp1[0] = 2 * phi[1, j, k, l, m, n] - phi[2, j, k, l, m, n]
                            tmp1[0] = my_max(tmp1[0], phi[2, j, k, l, m, n])
                            phi[0, j, k, l, m, n] = my_min(tmp1[0], phi[0, j, k, l, m, n])
                            phi[0, j, k, l, m, n] = my_max(phi[0, j, k, l, m, n], constraint[0, j, k, l, m, n])

                            tmp2 = hcl.scalar(0, "tmp2")
                            tmp2[0] = 2 * phi[phi.shape[0] - 2, j, k, l, m, n] - phi[phi.shape[0] - 3, j, k, l, m, n]
                            tmp2[0] = my_max(tmp2[0], phi[phi.shape[0] - 3, j, k, l, m, n])
                            phi[phi.shape[0] - 1, j, k, l, m, n] = my_min(tmp2[0], phi[phi.shape[0] - 1, j, k, l, m, n])
                            phi[phi.shape[0] - 1, j, k, l, m, n] = my_max(phi[phi.shape[0] - 1, j, k, l, m, n], constraint[phi.shape[0] - 1, j, k, l, m, n])

    if 1 not in g.pDim:
        with hcl.for_(0, phi.shape[0], name="i") as i:
            with hcl.for_(0, phi.shape[2], name="k") as k:
                with hcl.for_(0, phi.shape[3], name="l") as l:
                    with hcl.for_(0, phi.shape[4], name="m") as m:
                        with hcl.for_(0, phi.shape[5], name="n") as n:
                            tmp1 = hcl.scalar(0, "tmp1")
                            tmp1[0] = 2 * phi[i, 1, k, l, m, n] - phi[i, 2, k, l, m, n]
                            tmp1[0] = my_max(tmp1[0], phi[i, 2, k, l, m, n])
                            phi[i, 0, k, l, m, n] = my_min(tmp1[0], phi[i, 0, k, l, m, n])
                            phi[i, 0, k, l, m, n] = my_max(phi[i, 0, k, l, m, n], constraint[i, 0, k, l, m, n])

                            tmp2 = hcl.scalar(0, "tmp2")
                            tmp2[0] = 2 * phi[i, phi.shape[1] - 2, k, l, m, n] - phi[i, phi.shape[1] - 3, k, l, m, n]
                            tmp2[0] = my_max(tmp2[0], phi[i, phi.shape[1] - 3, k, l, m, n])
                            phi[i, phi.shape[1] - 1, k, l, m, n] = my_min(tmp2[0], phi[i, phi.shape[1] - 1, k, l, m, n])
                            phi[i, phi.shape[1] - 1, k, l, m, n] = my_max(phi[i, phi.shape[1] - 1, k, l, m, n], constraint[i, phi.shape[1] - 1, k, l, m, n])

    if 2 not in g.pDim:
        with hcl.for_(0, phi.shape[0], name="i") as i:
            with hcl.for_(0, phi.shape[1], name="j") as j:
                with hcl.for_(0, phi.shape[3], name="l") as l:
                    with hcl.for_(0, phi.shape[4], name="m") as m:
                        with hcl.for_(0, phi.shape[5], name="n") as n:
                            tmp1 = hcl.scalar(0, "tmp1")
                            tmp1[0] = 2 * phi[i, j, 1, l, m, n] - phi[i, j, 2, l, m, n]
                            tmp1[0] = my_max(tmp1[0], phi[i, j, 2, l, m, n])
                            phi[i, j, 0, l, m, n] = my_min(tmp1[0], phi[i, j, 0, l, m, n])
                            phi[i, j, 0, l, m, n] = my_max(phi[i, j, 0, l, m, n], constraint[i, j, 0, l, m, n])

                            tmp2 = hcl.scalar(0, "tmp2")
                            tmp2[0] = 2 * phi[i, j, phi.shape[2] - 2, l, m, n] - phi[i, j, phi.shape[2] - 3, l, m, n]
                            tmp2[0] = my_max(tmp2[0], phi[i, j, phi.shape[2] - 3, l, m, n])
                            phi[i, j, phi.shape[2] - 1, l, m, n] = my_min(tmp2[0], phi[i, j, phi.shape[2] - 1, l, m, n])
                            phi[i, j, phi.shape[2] - 1, l, m, n] = my_max(phi[i, j, phi.shape[2] - 1, l, m, n], constraint[i, j, phi.shape[2] - 1, l, m, n])
                            
    if 3 not in g.pDim:
        with hcl.for_(0, phi.shape[0], name="i") as i:
            with hcl.for_(0, phi.shape[1], name="j") as j:
                with hcl.for_(0, phi.shape[2], name="k") as k:
                    with hcl.for_(0, phi.shape[4], name="m") as m:
                        with hcl.for_(0, phi.shape[5], name="n") as n:
                            tmp1 = hcl.scalar(0, "tmp1")
                            tmp1[0] = 2 * phi[i, j, k, 1, m, n] - phi[i, j, k, 2, m, n]
                            tmp1[0] = my_max(tmp1[0], phi[i, j, k, 2, m, n])
                            phi[i, j, k, 0, m, n] = my_min(tmp1[0], phi[i, j, k, 0, m, n])
                            phi[i, j, k, 0, m, n] = my_max(phi[i, j, k, 0, m, n], constraint[i, j, k, 0, m, n])

                            tmp2 = hcl.scalar(0, "tmp2")
                            tmp2[0] = 2 * phi[i, j, k, phi.shape[3] - 2, m, n] - phi[i, j, k, phi.shape[3] - 3, m, n]
                            tmp2[0] = my_max(tmp2[0], phi[i, j, k, phi.shape[3] - 3, m, n])
                            phi[i, j, k, phi.shape[3] - 1, m, n] = my_min(tmp2[0], phi[i, j, k, phi.shape[3] - 1, m, n])
                            phi[i, j, k, phi.shape[3] - 1, m, n] = my_max(phi[i, j, k, phi.shape[3] - 1, m, n], constraint[i, j, k, phi.shape[3] - 1, m, n])

    if 4 not in g.pDim:
        with hcl.for_(0, phi.shape[0], name="i") as i:
            with hcl.for_(0, phi.shape[1], name="j") as j:
                with hcl.for_(0, phi.shape[2], name="k") as k:
                    with hcl.for_(0, phi.shape[3], name="l") as l:
                        with hcl.for_(0, phi.shape[5], name="n") as n:
                            tmp1 = hcl.scalar(0, "tmp1")
                            tmp1[0] = 2 * phi[i, j, k, l, 1, n] - phi[i, j, k, l, 2, n]
                            tmp1[0] = my_max(tmp1[0], phi[i, j, k, l, 2, n])
                            phi[i, j, k, l, 0, n] = my_min(tmp1[0], phi[i, j, k, l, 0, n])
                            phi[i, j, k, l, 0, n] = my_max(phi[i, j, k, l, 0, n], constraint[i, j, k, l, 0, n])

                            tmp2 = hcl.scalar(0, "tmp2")
                            tmp2[0] = 2 * phi[i, j, k, l, phi.shape[4] - 2, n] - phi[i, j, k, l, phi.shape[4] - 3, n]
                            tmp2[0] = my_max(tmp2[0], phi[i, j, k, l, phi.shape[4] - 3, n])
                            phi[i, j, k, l, phi.shape[4] - 1, n] = my_min(tmp2[0], phi[i, j, k, l, phi.shape[4] - 1, n])
                            phi[i, j, k, l, phi.shape[4] - 1, n] = my_max(phi[i, j, k, l, phi.shape[4] - 1, n], constraint[i, j, k, l, phi.shape[4] - 1, n])
    
    if 5 not in g.pDim:
        with hcl.for_(0, phi.shape[0], name="i") as i:
            with hcl.for_(0, phi.shape[1], name="j") as j:
                with hcl.for_(0, phi.shape[2], name="k") as k:
                    with hcl.for_(0, phi.shape[3], name="l") as l:
                        with hcl.for_(0, phi.shape[4], name="m") as m:
                            tmp1 = hcl.scalar(0, "tmp1")
                            tmp1[0] = 2 * phi[i, j, k, l, m, 1] - phi[i, j, k, l, m, 2]
                            tmp1[0] = my_max(tmp1[0], phi[i, j, k, l, m, 2])
                            phi[i, j, k, l, m, 0] = my_min(tmp1[0], phi[i, j, k, l, m, 0])
                            phi[i, j, k, l, m, 0] = my_max(phi[i, j, k, l, m, 0], constraint[i, j, k, l, m, 0])

                            tmp2 = hcl.scalar(0, "tmp2")
                            tmp2[0] = 2 * phi[i, j, k, l, m, phi.shape[5] - 2] - phi[i, j, k, l, m, phi.shape[5] - 3]
                            tmp2[0] = my_max(tmp2[0], phi[i, j, k, l, m, phi.shape[5] - 3])
                            phi[i, j, k, l, m, phi.shape[5] - 1] = my_min(tmp2[0], phi[i, j, k, l, m, phi.shape[5] - 1])
                            phi[i, j, k, l, m, phi.shape[5] - 1] = my_max(phi[i, j, k, l, m, phi.shape[5] - 1], constraint[i, j, k, l, m, phi.shape[5] - 1])

######################################### VALUE ITERATION ##########################################


def TTR_6D(my_object, g):
    def solve_phiNew(phi, constraint, x1, x2, x3, x4, x5, x6):
        l_i = 0 if 0 in g.pDim else 1
        h_i = phi.shape[0] if 0 in g.pDim else phi.shape[0] - 1
        l_j = 0 if 1 in g.pDim else 1
        h_j = phi.shape[1] if 1 in g.pDim else phi.shape[1] - 1
        l_k = 0 if 2 in g.pDim else 1
        h_k = phi.shape[2] if 2 in g.pDim else phi.shape[2] - 1
        l_l = 0 if 3 in g.pDim else 1
        h_l = phi.shape[3] if 3 in g.pDim else phi.shape[3] - 1
        l_m = 0 if 4 in g.pDim else 1
        h_m = phi.shape[4] if 4 in g.pDim else phi.shape[4] - 1
        l_n = 0 if 5 in g.pDim else 1
        h_n = phi.shape[5] if 5 in g.pDim else phi.shape[5] - 1

        # Perform value iteration by sweeping in direction 1
        with hcl.Stage("Sweep_1"):
            with hcl.for_(l_i, h_i, name="i") as i:
                with hcl.for_(l_j, h_j, name="j") as j:
                    with hcl.for_(l_k, h_k, name="k") as k:
                        with hcl.for_(l_l, h_l, name="l") as l:
                            with hcl.for_(l_m, h_m, name="m") as m:
                                with hcl.for_(l_n, h_n, name="n") as n:
                                    updatePhi(i, j, k, l, m, n, my_object, phi, constraint, g, x1, x2, x3, x4, x5, x6)
            EvalBoundary(phi, constraint, g)

        # Perform value iteration by sweeping in direction 2
        with hcl.Stage("Sweep_2"):
            with hcl.for_(l_i, h_i, name="i") as i:
                with hcl.for_(l_j, h_j, name="j") as j:
                    with hcl.for_(l_k, h_k, name="k") as k:
                        with hcl.for_(l_l, h_l, name="l") as l:
                            with hcl.for_(l_m, h_m, name="m") as m:
                                with hcl.for_(l_n, h_n, name="n") as n:
                                    i2 = phi.shape[0] - i - 1
                                    j2 = phi.shape[1] - j - 1
                                    k2 = phi.shape[2] - k - 1
                                    l2 = phi.shape[3] - l - 1
                                    m2 = phi.shape[4] - m - 1
                                    n2 = phi.shape[5] - n - 1
                                    updatePhi(i2, j2, k2, l2, m2, n2, my_object, phi, constraint, g, x1, x2, x3, x4, x5, x6)
            EvalBoundary(phi, constraint, g)

        # Perform value iteration by sweeping in direction 3
        with hcl.Stage("Sweep_3"):
            with hcl.for_(l_i, h_i, name="i") as i:
                with hcl.for_(l_j, h_j, name="j") as j:
                    with hcl.for_(l_k, h_k, name="k") as k:
                        with hcl.for_(l_l, h_l, name="l") as l:
                            with hcl.for_(l_m, h_m, name="m") as m:
                                with hcl.for_(l_n, h_n, name="n") as n:
                                    i2 = phi.shape[0] - i - 1
                                    j2 = phi.shape[1] - j - 1
                                    k2 = phi.shape[2] - k - 1
                                    l2 = phi.shape[3] - l - 1
                                    m2 = phi.shape[4] - m - 1
                                    updatePhi(i2, j2, k2, l2, m, n, my_object, phi, constraint, g, x1, x2, x3, x4, x5, x6)
            EvalBoundary(phi, constraint, g)

        # Perform value iteration by sweeping in direction 4
        with hcl.Stage("Sweep_4"):
            with hcl.for_(l_i, h_i, name="i") as i:
                with hcl.for_(l_j, h_j, name="j") as j:
                    with hcl.for_(l_k, h_k, name="k") as k:
                        with hcl.for_(l_l, h_l, name="l") as l:
                            with hcl.for_(l_m, h_m, name="m") as m:
                                with hcl.for_(l_n, h_n, name="n") as n:
                                    i2 = phi.shape[0] - i - 1
                                    j2 = phi.shape[1] - j - 1
                                    k2 = phi.shape[2] - k - 1
                                    l2 = phi.shape[3] - l - 1
                                    updatePhi(i2, j2, k2, l2, m, n, my_object, phi, constraint, g, x1, x2, x3, x4, x5, x6)
            EvalBoundary(phi, constraint, g)

        # Perform value iteration by sweeping in direction 5
        with hcl.Stage("Sweep_5"):
            with hcl.for_(l_i, h_i, name="i") as i:
                with hcl.for_(l_j, h_j, name="j") as j:
                    with hcl.for_(l_k, h_k, name="k") as k:
                        with hcl.for_(l_l, h_l, name="l") as l:
                            with hcl.for_(l_m, h_m, name="m") as m:
                                with hcl.for_(l_n, h_n, name="n") as n:
                                    i2 = phi.shape[0] - i - 1
                                    j2 = phi.shape[1] - j - 1
                                    k2 = phi.shape[2] - k - 1
                                    updatePhi(i2, j2, k2, l, m, n, my_object, phi, constraint, g, x1, x2, x3, x4, x5, x6)
            EvalBoundary(phi, constraint, g)

        # Perform value iteration by sweeping in direction 6
        with hcl.Stage("Sweep_6"):
            with hcl.for_(l_i, h_i, name="i") as i:
                with hcl.for_(l_j, h_j, name="j") as j:
                    with hcl.for_(l_k, h_k, name="k") as k:
                        with hcl.for_(l_l, h_l, name="l") as l:
                            with hcl.for_(l_m, h_m, name="m") as m:
                                with hcl.for_(l_n, h_n, name="n") as n:
                                    i2 = phi.shape[0] - i - 1
                                    j2 = phi.shape[1] - j - 1
                                    updatePhi(i2, j2, k, l, m, n, my_object, phi, constraint, g, x1, x2, x3, x4, x5, x6)
            EvalBoundary(phi, constraint, g)

        # Perform value iteration by sweeping in direction 7
        with hcl.Stage("Sweep_7"):
            with hcl.for_(l_i, h_i, name="i") as i:
                with hcl.for_(l_j, h_j, name="j") as j:
                    with hcl.for_(l_k, h_k, name="k") as k:
                        with hcl.for_(l_l, h_l, name="l") as l:
                            with hcl.for_(l_m, h_m, name="m") as m:
                                with hcl.for_(l_n, h_n, name="n") as n:
                                    i2 = phi.shape[0] - i - 1
                                    updatePhi(i2, j, k, l, m, n, my_object, phi, constraint, g, x1, x2, x3, x4, x5, x6)
            EvalBoundary(phi, constraint, g)

        # Perform value iteration by sweeping in direction 8
        with hcl.Stage("Sweep_8"):
            with hcl.for_(l_i, h_i, name="i") as i:
                with hcl.for_(l_j, h_j, name="j") as j:
                    with hcl.for_(l_k, h_k, name="k") as k:
                        with hcl.for_(l_l, h_l, name="l") as l:
                            with hcl.for_(l_m, h_m, name="m") as m:
                                with hcl.for_(l_n, h_n, name="n") as n:
                                    j2 = phi.shape[1] - j - 1
                                    updatePhi(i, j2, k, l, m, n, my_object, phi, constraint, g, x1, x2, x3, x4, x5, x6)
            EvalBoundary(phi, constraint, g)

    ###################################### SETUP PLACEHOLDERS ######################################
    x1 = hcl.placeholder((g.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((g.pts_each_dim[1],), name="x2", dtype=hcl.Float())
    x3 = hcl.placeholder((g.pts_each_dim[2],), name="x3", dtype=hcl.Float())
    x4 = hcl.placeholder((g.pts_each_dim[3],), name="x4", dtype=hcl.Float())
    x5 = hcl.placeholder((g.pts_each_dim[4],), name="x5", dtype=hcl.Float())
    x6 = hcl.placeholder((g.pts_each_dim[5],), name="x6", dtype=hcl.Float())
    phi = hcl.placeholder(tuple(g.pts_each_dim), name="phi", dtype=hcl.Float())
    constraint = hcl.placeholder(tuple(g.pts_each_dim), name="constraint", dtype=hcl.Float())
    # debugger = hcl.placeholder(tuple(g.pts_each_dim), name="debugger", dtype=hcl.Float())
    # debug2 = hcl.placeholder((0,), "debug2")
        # Create a static schedule -- graph
    s = hcl.create_schedule([phi, constraint, x1, x2, x3, x4, x5, x6], solve_phiNew)
    sweep_1 = solve_phiNew.Sweep_1
    sweep_2 = solve_phiNew.Sweep_2
    sweep_3 = solve_phiNew.Sweep_3
    sweep_4 = solve_phiNew.Sweep_4
    sweep_5 = solve_phiNew.Sweep_5
    sweep_6 = solve_phiNew.Sweep_6
    sweep_7 = solve_phiNew.Sweep_7
    sweep_8 = solve_phiNew.Sweep_8

    s[sweep_1].parallel(sweep_1.i)
    s[sweep_2].parallel(sweep_2.i)
    s[sweep_3].parallel(sweep_3.i)
    s[sweep_4].parallel(sweep_4.i)
    s[sweep_5].parallel(sweep_5.i)
    s[sweep_6].parallel(sweep_6.i)
    s[sweep_7].parallel(sweep_7.i)
    s[sweep_8].parallel(sweep_8.i)
    # Build an executable and return
    return hcl.build(s)
       