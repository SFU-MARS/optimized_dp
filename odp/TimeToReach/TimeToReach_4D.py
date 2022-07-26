import heterocl as hcl
import numpy as np
from odp.computeGraphs.graph_4D import *


# Update the phi function at position (i,j,k)
def updatePhi(i, j, k, l, my_object, phi, g, x1, x2, x3, x4, debugger):
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

    sigma1 = hcl.scalar(0, "sigma1")
    sigma2 = hcl.scalar(0, "sigma2")
    sigma3 = hcl.scalar(0, "sigma3")
    sigma4 = hcl.scalar(0, "sigma4")

    # dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
    dV_dx1_L[0], dV_dx1_R[0] = spa_derivX1_4d(i, j, k, l, phi, g)
    dV_dx2_L[0], dV_dx2_R[0] = spa_derivX2_4d(i, j, k, l, phi, g)
    dV_dx3_L[0], dV_dx3_R[0] = spa_derivX3_4d(i, j, k, l, phi, g)
    dV_dx4_L[0], dV_dx4_R[0] = spa_derivX4_4d(i, j, k, l, phi, g)

    # Calculate average gradient
    dV_dx1[0] = (dV_dx1_L[0] + dV_dx1_R[0]) / 2
    dV_dx2[0] = (dV_dx2_L[0] + dV_dx2_R[0]) / 2
    dV_dx3[0] = (dV_dx3_L[0] + dV_dx3_R[0]) / 2
    dV_dx4[0] = (dV_dx4_L[0] + dV_dx4_R[0]) / 2

    # Find the optimal control through my_object's API
    uOpt = my_object.opt_ctrl(0, (x1[i], x2[j], x3[k], x4[l]),
                              (dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0]))
    # Find optimal disturbance
    dOpt = my_object.opt_dstb((dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0]))

    # Find rates of changes based on dynamics equation
    dx1_dt, dx2_dt, dx3_dt, dx4_dt = my_object.dynamics(0, (x1[i], x2[j], x3[k], x4[l]), uOpt, dOpt)

    H = hcl.scalar(0, "H")
    phiNew = hcl.scalar(0, "phiNew")
    diss1 = hcl.scalar(0, "diss1")
    diss2 = hcl.scalar(0, "diss2")
    diss3 = hcl.scalar(0, "diss3")
    diss4 = hcl.scalar(0, "diss4")

    # Calculate Hamiltonian terms:
    H[0] = (-(dx1_dt * dV_dx1[0] + dx2_dt * dV_dx2[0] + dx3_dt * dV_dx3[0] + dx4_dt * dV_dx4[0] + 1))

    # Calculate the "dissipation"
    sigma1[0] = my_abs(dx1_dt)
    sigma2[0] = my_abs(dx2_dt)
    sigma3[0] = my_abs(dx3_dt)
    sigma4[0] = my_abs(dx4_dt)
    c = hcl.scalar(0, "c")
    c[0] = sigma1[0] / g.dx[0] + sigma2[0] / g.dx[1] + sigma3[0] / g.dx[2] + sigma4[0] / g.dx[3]

    diss1[0] = sigma1[0] * ((dV_dx1_R[0] - dV_dx1_L[0]) / 2 + phi[i, j, k, l] / g.dx[0])
    diss2[0] = sigma2[0] * ((dV_dx2_R[0] - dV_dx2_L[0]) / 2 + phi[i, j, k, l] / g.dx[1])
    diss3[0] = sigma3[0] * ((dV_dx3_R[0] - dV_dx3_L[0]) / 2 + phi[i, j, k, l] / g.dx[2])
    diss4[0] = sigma4[0] * ((dV_dx4_R[0] - dV_dx4_L[0]) / 2 + phi[i, j, k, l] / g.dx[3])

    # New phi
    phiNew[0] = (-H[0] + diss1[0] + diss2[0] + diss3[0] + diss4[0]) / c[0]
    debugger[i, j, k, l] = phiNew[0]
    phi[i, j, k, l] = my_min(phi[i, j, k, l], phiNew[0])

def EvalBoundary(phi, g, debug2):
    if 0 not in g.pDim:
        with hcl.for_(0, phi.shape[1], name="j") as j:
            with hcl.for_(0, phi.shape[2], name="k") as k:
                with hcl.for_(0, phi.shape[3], name="l") as l:
                    debug2[0] = j
                    tmp1 = hcl.scalar(0, "tmp1")
                    tmp1[0] = 2 * phi[1, j, k, l] - phi[2, j, k, l]
                    tmp1[0] = my_max(tmp1[0], phi[2, j, k, l])
                    phi[0, j, k, l] = my_min(tmp1[0], phi[0, j, k, l])

                    tmp2 = hcl.scalar(0, "tmp2")
                    tmp2[0] = 2 * phi[phi.shape[0] - 2, j, k, l] - phi[phi.shape[0] - 3, j, k, l]
                    tmp2[0] = my_max(tmp2[0], phi[phi.shape[0] - 3, j, k, l])
                    phi[phi.shape[0] - 1, j, k, l] = my_min(tmp2[0], phi[phi.shape[0] - 1, j, k, l])

    if 1 not in g.pDim:
        with hcl.for_(0, phi.shape[0], name="i") as i2:
            with hcl.for_(0, phi.shape[2], name="k") as k:
                with hcl.for_(0, phi.shape[3], name="l") as l:
                    tmp1 = hcl.scalar(0, "tmp1")
                    tmp1[0] = 2 * phi[i2, 1, k, l] - phi[i2, 2, k, l]
                    tmp1[0] = my_max(tmp1[0], phi[i2, 2, k, l ])
                    phi[i2, 0, k, l] = my_min(tmp1[0], phi[i2, 0, k, l])

                    tmp2 = hcl.scalar(0, "tmp2")
                    tmp2[0] = 2 * phi[i2, phi.shape[1] - 2, k, l] - phi[i2, phi.shape[1] - 3, k, l]
                    tmp2[0] = my_max(tmp2[0], phi[i2, phi.shape[1] - 3, k, l])
                    phi[i2, phi.shape[1] - 1, k, l] = my_min(tmp2[0], phi[i2, phi.shape[1] - 1, k, l])

    if 2 not in g.pDim:
        with hcl.for_(0, phi.shape[0], name="i") as i2:
            with hcl.for_(0, phi.shape[1], name="j") as j:
                with hcl.for_(0, phi.shape[3], name="l") as l:
                    tmp1 = hcl.scalar(0, "tmp1")
                    tmp1[0] = 2 * phi[i2, j, 1, l] - phi[i2, j, 2, l]
                    tmp1[0] = my_max(tmp1[0], phi[i2, j, 2, l])
                    phi[i2, j, 0, l] = my_min(tmp1[0], phi[i2, j, 0, l])

                    tmp2 = hcl.scalar(0, "tmp2")
                    tmp2[0] = 2 * phi[i2, j, phi.shape[2] - 2, l] - phi[i2, j, phi.shape[2] - 3, l]
                    tmp2[0] = my_max(tmp2[0], phi[i2, j, phi.shape[2] - 3, l])
                    phi[i2, j, phi.shape[2] - 1, l] = my_min(tmp2[0], phi[i2, j, phi.shape[2] - 1, l])

    if 3 not in g.pDim:
        with hcl.for_(0, phi.shape[0], name="i") as i2:
            with hcl.for_(0, phi.shape[1], name="j") as j:
                with hcl.for_(0, phi.shape[2], name="k") as k:
                    tmp1 = hcl.scalar(0, "tmp1")
                    tmp1[0] = 2 * phi[i2, j, k, 1] - phi[i2, j, k, 2]
                    tmp1[0] = my_max(tmp1[0], phi[i2, j, k, 2])
                    phi[i2, j, k, 0] = my_min(tmp1[0], phi[i2, j, k, 0])

                    tmp2 = hcl.scalar(0, "tmp2")
                    tmp2[0] = 2 * phi[i2, j, k, phi.shape[2] - 2] - phi[i2, j, k, phi.shape[2] - 3]
                    tmp2[0] = my_max(tmp2[0], phi[i2, j, k, phi.shape[2] - 3])
                    phi[i2, j, k, phi.shape[2] - 1] = my_min(tmp2[0], phi[i2, j, k, phi.shape[2] - 1])



######################################### VALUE ITERATION ##########################################

def TTR_4D(my_object, g):
    def solve_phiNew(phi, x1, x2, x3, x4, debugger, debug2):
            l_i = 0 if 0 in g.pDim else 1
            h_i = phi.shape[0] if 0 in g.pDim else phi.shape[0] - 1
            l_j = 0 if 1 in g.pDim else 1
            h_j = phi.shape[1] if 1 in g.pDim else phi.shape[1] - 1
            l_k = 0 if 2 in g.pDim else 1
            h_k = phi.shape[2] if 2 in g.pDim else phi.shape[2] - 1
            l_l = 0 if 3 in g.pDim else 1
            h_l = phi.shape[3] if 3 in g.pDim else phi.shape[3] - 1
            # Perform value iteration by sweeping in direction 1
            with hcl.Stage("Sweep_1"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        with hcl.for_(l_k, h_k, name="k") as k:
                            with hcl.for_(l_l, h_l, name="l") as l:
                                updatePhi(i, j, k, l, my_object, phi, g, x1, x2, x3, x4, debugger)
                EvalBoundary(phi, g, debug2)

            # Perform value iteration by sweeping in direction 2
            with hcl.Stage("Sweep_2"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        with hcl.for_(l_k, h_k, name="k") as k:
                            with hcl.for_(l_l, h_l, name="l") as l:
                                i2 = phi.shape[0] - i - 1
                                j2 = phi.shape[1] - j - 1
                                k2 = phi.shape[2] - k - 1
                                updatePhi(i2, j2, k2, l, my_object, phi, g, x1, x2, x3, x4, debugger)
                EvalBoundary(phi, g, debug2)

            # Perform value iteration by sweeping in direction 3
            with hcl.Stage("Sweep_3"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        with hcl.for_(l_k, h_k, name="k") as k:
                            with hcl.for_(l_l, h_l, name="l") as l:
                                j2 = phi.shape[1] - j - 1
                                k2 = phi.shape[2] - k - 1
                                updatePhi(i, j2, k2, l, my_object, phi, g, x1, x2, x3, x4, debugger)
                EvalBoundary(phi, g, debug2)

            # Perform value iteration by sweeping in direction 4
            with hcl.Stage("Sweep_4"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        with hcl.for_(l_k, h_k, name="k") as k:
                            with hcl.for_(l_l, h_l, name="l") as l:
                                i2 = phi.shape[0] - i - 1
                                j2 = phi.shape[1] - j - 1
                                updatePhi(i2, j2, k, l, my_object, phi, g, x1, x2, x3, x4, debugger)
                EvalBoundary(phi, g, debug2)

            # Perform value iteration by sweeping in direction 5
            with hcl.Stage("Sweep_5"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        with hcl.for_(l_k, h_k, name="k") as k:
                            with hcl.for_(l_l, h_l, name="l") as l:
                                i2 = phi.shape[0] - i - 1
                                l2 = phi.shape[3] - l - 1
                                updatePhi(i2, j, k, l2, my_object, phi, g, x1, x2, x3, x4, debugger)
                EvalBoundary(phi, g, debug2)

            # Perform value iteration by sweeping in direction 6
            with hcl.Stage("Sweep_6"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        with hcl.for_(l_k, h_k, name="k") as k:
                            with hcl.for_(l_l, h_l, name="l") as l:
                                j2 = phi.shape[1] - j - 1
                                l2 = phi.shape[3] - l - 1
                                updatePhi(i, j2, k, l2, my_object, phi, g, x1, x2, x3, x4, debugger)
                EvalBoundary(phi, g, debug2)

            # Perform value iteration by sweeping in direction 7
            with hcl.Stage("Sweep_7"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        with hcl.for_(l_k, h_k, name="k") as k:
                            with hcl.for_(l_l, h_l, name="l") as l:
                                k2 = phi.shape[2] - k - 1
                                l2 = phi.shape[3] - l - 1
                                updatePhi(i, j, k2, l2, my_object, phi, g, x1, x2, x3, x4, debugger)
                EvalBoundary(phi, g, debug2)

            # Perform value iteration by sweeping in direction 8
            with hcl.Stage("Sweep_8"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        with hcl.for_(l_k, h_k, name="k") as k:
                            with hcl.for_(l_l, h_l, name="l") as l:
                                i2 = phi.shape[0] - i - 1
                                j2 = phi.shape[1] - j - 1
                                k2 = phi.shape[2] - k - 1
                                l2 = phi.shape[3] - l - 1
                                updatePhi(i2, j2, k2, l2, my_object, phi, g, x1, x2, x3, x4, debugger)
                EvalBoundary(phi, g, debug2)


    ###################################### SETUP PLACEHOLDERS ######################################

    # Positions vector
    x1 = hcl.placeholder((g.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((g.pts_each_dim[1],), name="x2", dtype=hcl.Float())
    x3 = hcl.placeholder((g.pts_each_dim[2],), name="x3", dtype=hcl.Float())
    x4 = hcl.placeholder((g.pts_each_dim[3],), name="x4", dtype=hcl.Float())
    phi = hcl.placeholder(tuple(g.pts_each_dim), name="phi", dtype=hcl.Float())
    debugger = hcl.placeholder(tuple(g.pts_each_dim), name="debugger", dtype=hcl.Float())
    debug2 = hcl.placeholder((0,), "debug2")

    # Create a static schedule -- graph
    s = hcl.create_schedule([phi, x1, x2, x3, x4, debugger, debug2], solve_phiNew)

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
