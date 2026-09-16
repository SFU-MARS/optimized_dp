import heterocl as hcl
import numpy as np
from odp.computeGraphs.graph_1D import *

# Update the phi function at position (i)
def updatePhi(i, my_object, phi, g, x1):
    dV_dx1_L = hcl.scalar(0, "dV_dx1_L")
    dV_dx1_R = hcl.scalar(0, "dV_dx1_R")
    dV_dx1 = hcl.scalar(0, "dV_dx1")

    sigma1 = hcl.scalar(0, "sigma1")

    # dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
    dV_dx1_L[0], dV_dx1_R[0] = spa_derivX(i, phi, g)

    # Calculate average gradient
    dV_dx1[0] = (dV_dx1_L[0] + dV_dx1_R[0]) / 2

    # Find the optimal control through my_object's API
    uOpt = my_object.opt_ctrl(0, (x1[i],),
                              (dV_dx1[0],))
    dOpt = my_object.opt_dstb(0, (x1[i],),
                             (dV_dx1[0],))

    # Calculate dynamical rates of changes
    dx1_dt = my_object.dynamics(0, (x1[i],), uOpt, dOpt)

    H = hcl.scalar(0, "H")
    phiNew = hcl.scalar(0, "phiNew")
    diss1 = hcl.scalar(0, "diss1")

    # Calculate Hamiltonian terms:
    H[0] = (-(dx1_dt[0] * dV_dx1[0] + 1))

    # Calculate the "dissipation"
    sigma1[0] = my_abs(dx1_dt[0])
    c = hcl.scalar(0, "c")
    c[0] = sigma1[0] / g.dx[0]
    diss1[0] = sigma1[0] * ((dV_dx1_R[0] - dV_dx1_L[0]) / 2 + phi[i] / g.dx[0])

    # New phi
    phiNew[0] = (-H[0] + diss1[0]) / c[0]
    phi[i] = my_min(phi[i], phiNew[0])

def EvalBoundary(phi, g):
    if 0 not in g.pDim:
        tmp1 = hcl.scalar(0, "tmp1")
        tmp1[0] = 2 * phi[1] - phi[2]
        tmp1[0] = my_max(tmp1[0], phi[2])
        phi[0] = my_min(tmp1[0], phi[0])

        tmp2 = hcl.scalar(0, "tmp2")
        tmp2[0] = 2 * phi[phi.shape[0] - 2] - phi[phi.shape[0] - 3]
        tmp2[0] = my_max(tmp2[0], phi[phi.shape[0] - 3])
        phi[phi.shape[0] - 1] = my_min(tmp2[0], phi[phi.shape[0] - 1])


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

def TTR_1D(my_object, g):
    def solve_phiNew(phi, x1):
            l_i = 0 if 0 in g.pDim else 1
            h_i = phi.shape[0] if 0 in g.pDim else phi.shape[0] - 1
            # Perform value iteration by sweeping in direction 1
            with hcl.Stage("Sweep_1"):
                with hcl.for_(l_i, h_i, name="i") as i:
                    updatePhi(i, my_object, phi, g, x1)
                EvalBoundary(phi, g)

    ###################################### SETUP PLACEHOLDERS ######################################
    
    # Initialize the HCL environment
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    # Positions vector
    x1 = hcl.placeholder((g.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    phi      = hcl.placeholder(tuple(g.pts_each_dim), name="phi", dtype=hcl.Float())

    # Create a static schedule -- graph
    s = hcl.create_schedule([phi, x1], solve_phiNew)
    sweep_1 = solve_phiNew.Sweep_1

    s[sweep_1].parallel(sweep_1.i)
    

    # Build an executable and return
    return hcl.build(s)
