import heterocl as hcl
import numpy as np
from odp.computeGraphs.graph_2D import *

# Update the phi function at position (i,j)
def updatePhi(i, j, my_object, phi, constraint, g, x1, x2):
    dV_dx1_L = hcl.scalar(0, "dV_dx1_L")
    dV_dx1_R = hcl.scalar(0, "dV_dx1_R")
    dV_dx1 = hcl.scalar(0, "dV_dx1")
    dV_dx2_L = hcl.scalar(0, "dV_dx2_L")
    dV_dx2_R = hcl.scalar(0, "dV_dx2_R")
    dV_dx2 = hcl.scalar(0, "dV_dx2")

    sigma1 = hcl.scalar(0, "sigma1")
    sigma2 = hcl.scalar(0, "sigma2")

    # dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
    dV_dx1_L[0], dV_dx1_R[0] = spa_derivX(i, j, phi, g)
    dV_dx2_L[0], dV_dx2_R[0] = spa_derivY(i, j, phi, g)

    # Calculate average gradient
    dV_dx1[0] = (dV_dx1_L[0] + dV_dx1_R[0]) / 2
    dV_dx2[0] = (dV_dx2_L[0] + dV_dx2_R[0]) / 2

    # Find the optimal control through my_object's API
    uOpt = my_object.opt_ctrl(0, (x1[i], x2[j]),
                              (dV_dx1[0], dV_dx2[0]))
    dOpt = my_object.opt_dstb(0, (x1[i], x2[j]),
                             (dV_dx1[0], dV_dx2[0]))

    # Calculate dynamical rates of changes
    dx1_dt, dx2_dt = my_object.dynamics(0, (x1[i], x2[j]), uOpt, dOpt)

    H = hcl.scalar(0, "H")
    phiNew = hcl.scalar(0, "phiNew")
    diss1 = hcl.scalar(0, "diss1")
    diss2 = hcl.scalar(0, "diss2")

    # Calculate Hamiltonian terms:
    H[0] = (-(dx1_dt * dV_dx1[0] + dx2_dt * dV_dx2[0] + 1))

    # Calculate the "dissipation"
    sigma1[0] = my_abs(dx1_dt)
    sigma2[0] = my_abs(dx2_dt)
    c = hcl.scalar(0, "c")
    c[0] = sigma1[0] / g.dx[0] + sigma2[0] / g.dx[1]
    diss1[0] = sigma1[0] * ((dV_dx1_R[0] - dV_dx1_L[0]) / 2 + phi[i, j] / g.dx[0])
    diss2[0] = sigma2[0] * ((dV_dx2_R[0] - dV_dx2_L[0]) / 2 + phi[i, j] / g.dx[1])

    # New phi
    phiNew[0] = (-H[0] + diss1[0] + diss2[0]) / c[0]
    #debugger[i,j,k] = phiNew[0]
    phi[i, j] = my_min(phi[i, j], phiNew[0])
    # Check with the obstacle
    phi[i, j] = my_max(phi[i, j], constraint[i, j])

def EvalBoundary(phi, constraint, g):
    if 0 not in g.pDim:
        with hcl.for_(0, phi.shape[1], name="j") as j:
                #debug2[0] = j
                tmp1 = hcl.scalar(0, "tmp1")
                tmp1[0] = 2 * phi[1, j] - phi[2, j]
                tmp1[0] = my_max(tmp1[0], phi[2, j])
                phi[0, j] = my_min(tmp1[0], phi[0, j])
                phi[0, j] = my_max(phi[0, j], constraint[0, j])

                tmp2 = hcl.scalar(0, "tmp2")
                tmp2[0] = 2 * phi[phi.shape[0] - 2, j] - phi[phi.shape[0] - 3, j]
                tmp2[0] = my_max(tmp2[0], phi[phi.shape[0] - 3, j])
                phi[phi.shape[0] - 1, j] = my_min(tmp2[0], phi[phi.shape[0] - 1, j])
                phi[phi.shape[0] - 1, j] = my_max(phi[phi.shape[0] - 1, j], constraint[phi.shape[0] - 1, j])

    if 1 not in g.pDim:
        with hcl.for_(0, phi.shape[0], name="i1") as i1:
                tmp1 = hcl.scalar(0, "tmp1")
                tmp1[0] = 2 * phi[i1, 1] - phi[i1, 2]
                tmp1[0] = my_max(tmp1[0], phi[i1, 2])
                phi[i1, 0] = my_min(tmp1[0], phi[i1, 0])
                phi[i1, 0] = my_max(phi[i1, 0], constraint[i1, 0])

                tmp2 = hcl.scalar(0, "tmp2")
                tmp2[0] = 2 * phi[i1, phi.shape[1] - 2] - phi[i1, phi.shape[1] - 3]
                tmp2[0] = my_max(tmp2[0], phi[i1, phi.shape[1] - 3])
                phi[i1, phi.shape[1] - 1] = my_min(tmp2[0], phi[i1, phi.shape[1] - 1])
                phi[i1, phi.shape[1] - 1] = my_max(phi[i1, phi.shape[1] - 1], constraint[i1, phi.shape[1] - 1])


# Returns 0 if convergence has been reached
def evaluateConvergence(newV, oldV, epsilon, reSweep):
    delta = hcl.scalar(0, "delta")
    # Calculate the difference, if it's negative, make it positive
    delta[0] = newV[0] - oldV[0]
    with hcl.if_(delta[0] < 0):
        delta[0] = delta[0] * -1
    with hcl.if_(delta[0] > epsilon[0]):
        reSweep[0] = 1

######################################### TIME-TO-REACH COMPUTATION ##########################################

def TTR_2D(my_object, g):
    def solve_phiNew(phi, constraint, x1, x2):
            l_i = 0 if 0 in g.pDim else 1
            h_i = phi.shape[0] if 0 in g.pDim else phi.shape[0] - 1
            l_j = 0 if 1 in g.pDim else 1
            h_j = phi.shape[1] if 1 in g.pDim else phi.shape[1] - 1
            # Perform value iteration by sweeping in direction 1
            with hcl.Stage("Sweep_1"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        updatePhi(i, j, my_object, phi, constraint, g, x1, x2)
                            # debug2[0] = j
                EvalBoundary(phi, constraint, g)

            #Perform value iteration by sweeping in direction 2
            with hcl.Stage("Sweep_2"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        i2 = phi.shape[0] - i - 1
                        j2 = phi.shape[1] - j - 1
                        updatePhi(i2, j2, my_object, phi, constraint, g, x1, x2)
                EvalBoundary(phi, constraint, g)

            # Perform value iteration by sweeping in direction 3
            with hcl.Stage("Sweep_3"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        j2 = phi.shape[1] - j - 1
                        updatePhi(i, j2, my_object, phi, constraint, g, x1, x2)
                EvalBoundary(phi, constraint, g)


            # Perform value iteration by sweeping in direction 4
            with hcl.Stage("Sweep_4"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        i2 = phi.shape[0] - i - 1
                        j2 = phi.shape[1] - j - 1
                        updatePhi(i2, j2, my_object, phi, constraint, g, x1, x2)
                EvalBoundary(phi, constraint, g)

            # Perform value iteration by sweeping in direction 5
            with hcl.Stage("Sweep_5"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        i2 = phi.shape[0] - i - 1
                        updatePhi(i2, j, my_object, phi, constraint, g, x1, x2)
                EvalBoundary(phi, constraint, g)


            # Perform value iteration by sweeping in direction 6
            with hcl.Stage("Sweep_6"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        i2 = phi.shape[0] - i - 1
                        updatePhi(i2, j, my_object, phi, constraint, g, x1, x2)
                EvalBoundary(phi, constraint, g)

            # Perform value iteration by sweeping in direction 7
            with hcl.Stage("Sweep_7"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        j2 = phi.shape[1] - j - 1
                        updatePhi(i, j2, my_object, phi, constraint, g, x1, x2)
                EvalBoundary(phi, constraint, g)

            # Perform value iteration by sweeping in direction 8
            with hcl.Stage("Sweep_8"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    with hcl.for_(l_j, h_j, name="j") as j:
                        updatePhi(i, j, my_object, phi, constraint, g, x1, x2)
                EvalBoundary(phi, constraint, g)

    ###################################### SETUP PLACEHOLDERS ######################################
    
    # Initialize the HCL environment
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    # Positions vector
    x1 = hcl.placeholder((g.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((g.pts_each_dim[1],), name="x2", dtype=hcl.Float())
    phi      = hcl.placeholder(tuple(g.pts_each_dim), name="phi", dtype=hcl.Float())
    constraint = hcl.placeholder(tuple(g.pts_each_dim), name="constraint", dtype=hcl.Float())
    #debugger = hcl.placeholder(tuple(g.pts_each_dim), name="debugger", dtype=hcl.Float())
    #debug2 = hcl.placeholder((0,), "debug2")

    # Create a static schedule -- graph
    s = hcl.create_schedule([phi, constraint, x1, x2], solve_phiNew)
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
