import heterocl as hcl
import numpy as np
import time
import odp.valueIteration.user_definer_3D_Q as UD



###################################### USER-DEFINED FUNCTIONS ######################################


# Given state and action, return successor states and their probabilities
# sVals:  the coordinates of state
# bounds: the lower and upper limits of the state space in each dimension
# trans:  holds each successor state and the probability of reaching that state
def transition(sVals, action, bounds, trans, goal):
    dx  = hcl.scalar(0, "dx")
    dy  = hcl.scalar(0, "dy")
    mag = hcl.scalar(0, "mag")

    # Check if moving from a goal state
    dx[0]  = sVals[0] - goal[0,0]
    dy[0]  = sVals[1] - goal[0,1]
    mag[0] = hcl.sqrt((dx[0] * dx[0]) + (dy[0] * dy[0]))
    with hcl.if_(hcl.and_(mag[0] <= 1.0, sVals[2] <= goal[1,1], sVals[2] >= goal[1,0])):
        trans[0, 0] = 0
    # Check if moving from an obstacle 
    with hcl.elif_(hcl.or_(sVals[0] < bounds[0,0] + 0.2, sVals[0] > bounds[0,1] - 0.2)):
        trans[0, 0] = 0
    with hcl.elif_(hcl.or_(sVals[1] < bounds[1,0] + 0.2, sVals[1] > bounds[1,1] - 0.2)):
        trans[0, 0] = 0
    # Standard move
    with hcl.else_():
        trans[0, 0] = 1.0
        trans[0, 1] = sVals[0] + (0.6 * action[0] * hcl.cos(sVals[2]))
        trans[0, 2] = sVals[1] + (0.6 * action[0] * hcl.sin(sVals[2]))
        trans[0, 3] = sVals[2] + (0.6 * action[1])
        # Adjust for periodic dimension
        with hcl.while_(trans[0, 3] > 3.141592653589793):
            trans[0, 3] -= 6.283185307179586
        with hcl.while_(trans[0, 3] < -3.141592653589793):
            trans[0, 3] += 6.283185307179586


# Return the reward for taking action from state
def reward(sVals, action, bounds, goal, trans):
    dx  = hcl.scalar(0, "dx")
    dy  = hcl.scalar(0, "dy")
    mag = hcl.scalar(0, "mag")
    rwd = hcl.scalar(0, "rwd")

    # Check if moving from a collision state, if so, assign a penalty
    with hcl.if_(hcl.or_(sVals[0] < bounds[0,0] + 0.2, sVals[0] > bounds[0,1] - 0.2)):
        rwd[0] = -400
    with hcl.elif_(hcl.or_(sVals[1] < bounds[1,0] + 0.2, sVals[1] > bounds[1,1] - 0.2)):
        rwd[0] = -400
    with hcl.else_():
        # Check if moving from a goal state
        dx[0]  = sVals[0] - goal[0,0]
        dy[0]  = sVals[1] - goal[0,1]
        mag[0] = hcl.sqrt((dx[0] * dx[0]) + (dy[0] * dy[0]))
        with hcl.if_(hcl.and_(mag[0] <= 1.0, sVals[2] <= goal[1,1], sVals[2] >= goal[1,0])):
            rwd[0] = 1000
        # Standard move
        with hcl.else_():
            rwd[0] = 0
    return rwd[0]



######################################### HELPER FUNCTIONS #########################################


# Update the value function at position (i,j,k)
def updateQopt(i, j, k, a, iVals, sVals, Qopt, actions, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal):
    r = hcl.scalar(0, "r")
    p = hcl.scalar(0, "p")
    # set iVals equal to (i,j,k) and sVals equal to the corresponding state values at (i,j,k)
    updateStateVals(i, j, k, iVals, sVals, bounds, ptsEachDim)
    # call the transition function to obtain the outcome(s) of action a from state (si,sj,sk)
    transition(sVals, actions[a], bounds, trans, goal)
    # initialize Qopt[i,j,k,a] with the immediate reward
    r[0]          = reward(sVals, actions[a], bounds, goal, trans)
    Qopt[i,j,k,a] = r[0]
    # maximize over successor Q-values
    with hcl.for_(0, trans.shape[0], name="si") as si:
        p[0]     = trans[si,0]
        sVals[0] = trans[si,1]
        sVals[1] = trans[si,2]
        sVals[2] = trans[si,3]
        # Nearest neighbour
        with hcl.if_(useNN[0] == 1):
            # obtain the nearest neighbour successor state
            stateToIndex(sVals, iVals, bounds, ptsEachDim)
            # maximize over successor state Q-values
            with hcl.if_(hcl.and_(iVals[0] < Qopt.shape[0], iVals[1] < Qopt.shape[1], iVals[2] < Qopt.shape[2])):
                with hcl.if_(hcl.and_(iVals[0] >= 0, iVals[1] >= 0, iVals[2] >= 0)):
                    with hcl.for_(0, actions.shape[0], name="a_") as a_:
                        with hcl.if_((r[0] + (gamma[0] * (p[0] * Qopt[iVals[0],iVals[1],iVals[2],a_]))) > Qopt[i,j,k,a]):
                            Qopt[i,j,k,a] = r[0] + (gamma[0] * (p[0] * Qopt[iVals[0],iVals[1],iVals[2],a_]))
        # Linear interpolation
        with hcl.if_(useNN[0] == 0):
            with hcl.if_(hcl.and_(sVals[0] <= bounds[0,1], sVals[1] <= bounds[1,1], sVals[2] <= bounds[2,1])):
                with hcl.if_(hcl.and_(sVals[0] >= bounds[0,0], sVals[1] >= bounds[1,0], sVals[2] >= bounds[2,0])):
                    stateToIndexInterpolants(Qopt, sVals, actions, bounds, ptsEachDim, interpV, fillVal)
                    Qopt[i,j,k,a] += (gamma[0] * (p[0] * interpV[0]))
        r[0] += Qopt[i,j,k,a]

# Returns 0 if convergence has been reached
def evaluateConvergence(newQ, oldQ, epsilon, reSweep):
    delta = hcl.scalar(0, "delta")
    # Calculate the difference, if it's negative, make it positive
    delta[0] = newQ[0] - oldQ[0]
    with hcl.if_(delta[0] < 0):
        delta[0] = delta[0] * -1
    with hcl.if_(delta[0] > epsilon[0]):
        reSweep[0] = 1


# convert state values into indeces using nearest neighbour
# NOTE: have to modify this to work with modular values
def stateToIndex(sVals, iVals, bounds, ptsEachDim):
    iVals[0] = ((sVals[0] - bounds[0,0]) / (bounds[0,1] - bounds[0,0])) *  (ptsEachDim[0] - 1)
    iVals[1] = ((sVals[1] - bounds[1,0]) / (bounds[1,1] - bounds[1,0])) *  (ptsEachDim[1] - 1)
    iVals[2] = ((sVals[2] - bounds[2,0]) / (bounds[2,1] - bounds[2,0])) *  (ptsEachDim[2] - 1)
    # NOTE: add 0.5 to simulate rounding
    iVals[0] = hcl.cast(hcl.Int(), iVals[0] + 0.5)
    iVals[1] = hcl.cast(hcl.Int(), iVals[1] + 0.5)
    iVals[2] = hcl.cast(hcl.Int(), iVals[2] + 0.5)


# convert indices into state values
def indexToState(iVals, sVals, bounds, ptsEachDim): 
    sVals[0] = bounds[0,0] + ( (bounds[0,1] - bounds[0,0]) * (iVals[0] / (ptsEachDim[0]-1)) ) 
    sVals[1] = bounds[1,0] + ( (bounds[1,1] - bounds[1,0]) * (iVals[1] / (ptsEachDim[1]-1)) ) 
    sVals[2] = bounds[2,0] + ( (bounds[2,1] - bounds[2,0]) * (iVals[2] / (ptsEachDim[2]-1)) )


# set iVals equal to (i,j,k) and sVals equal to the corresponding state values at (i,j,k)
def updateStateVals(i, j, k, iVals, sVals, bounds, ptsEachDim):
    iVals[0] = i
    iVals[1] = j
    iVals[2] = k
    indexToState(iVals, sVals, bounds, ptsEachDim)


# given state values sVals, obtain the 8 possible successor states and their corresponding weight
def stateToIndexInterpolants(Qopt, sVals, actions, bounds, ptsEachDim, interpV, fillVal):
    iMin = hcl.scalar(0, "iMin")
    jMin = hcl.scalar(0, "jMin")
    kMin = hcl.scalar(0, "kMin")
    iMax = hcl.scalar(0, "iMax")
    jMax = hcl.scalar(0, "jMax")
    kMax = hcl.scalar(0, "kMax")
    c000 = hcl.scalar(fillVal[0], "c000")
    c001 = hcl.scalar(fillVal[0], "c001")
    c010 = hcl.scalar(fillVal[0], "c010")
    c011 = hcl.scalar(fillVal[0], "c011")
    c100 = hcl.scalar(fillVal[0], "c100")
    c101 = hcl.scalar(fillVal[0], "c101")
    c110 = hcl.scalar(fillVal[0], "c110")
    c111 = hcl.scalar(fillVal[0], "c111")
    c00  = hcl.scalar(0, "c00")
    c01  = hcl.scalar(0, "c01")
    c10  = hcl.scalar(0, "c10")
    c11  = hcl.scalar(0, "c11")
    c0   = hcl.scalar(0, "c0") 
    c1   = hcl.scalar(0, "c1") 
    ia   = hcl.scalar(0, "ia")
    ja   = hcl.scalar(0, "ja")
    ka   = hcl.scalar(0, "ka")
    di   = hcl.scalar(0, "di")
    dj   = hcl.scalar(0, "dj")
    dk   = hcl.scalar(0, "dk")

    # obtain unrounded index values
    ia[0] = ((sVals[0] - bounds[0,0]) / (bounds[0,1] - bounds[0,0])) *  (ptsEachDim[0] - 1)
    ja[0] = ((sVals[1] - bounds[1,0]) / (bounds[1,1] - bounds[1,0])) *  (ptsEachDim[1] - 1)
    ka[0] = ((sVals[2] - bounds[2,0]) / (bounds[2,1] - bounds[2,0])) *  (ptsEachDim[2] - 1)

    # obtain neighbouring state indeces in each direction
    with hcl.if_(ia[0] < 0):
        iMin[0] = hcl.cast(hcl.Int(), ia[0] - 1.0)
        iMax[0] = hcl.cast(hcl.Int(), ia[0])
    with hcl.else_():
        iMin[0] = hcl.cast(hcl.Int(), ia[0])
        iMax[0] = hcl.cast(hcl.Int(), ia[0] + 1.0)
    with hcl.if_(ja[0] < 0):
        jMin[0] = hcl.cast(hcl.Int(), ja[0] - 1.0)
        jMax[0] = hcl.cast(hcl.Int(), ja[0])
    with hcl.else_():
        jMin[0] = hcl.cast(hcl.Int(), ja[0])
        jMax[0] = hcl.cast(hcl.Int(), ja[0] + 1.0)
    with hcl.if_(ka[0] < 0):
        kMin[0] = hcl.cast(hcl.Int(), ka[0] - 1.0)
        kMax[0] = hcl.cast(hcl.Int(), ka[0]) 
    with hcl.else_():
        kMin[0] = hcl.cast(hcl.Int(), ka[0])
        kMax[0] = hcl.cast(hcl.Int(), ka[0] + 1.0) 

    # obtain weights in each direction
    di[0] = ia[0] - iMin[0]
    dj[0] = ja[0] - jMin[0]
    dk[0] = ka[0] - kMin[0]

    # Obtain value of each neighbour state
    # Qopt[iMin, jMin, kMin]
    with hcl.if_(hcl.and_(iMin[0] < Qopt.shape[0], jMin[0] < Qopt.shape[1], kMin[0] < Qopt.shape[2])):
        with hcl.if_(hcl.and_(iMin[0] >= 0, jMin[0] >= 0, kMin[0] >= 0)):
            with hcl.for_(0, actions.shape[0], name="a_") as a_:
                with hcl.if_(c000[0] < Qopt[iMin[0], jMin[0], kMin[0], a_]):
                    c000[0] = Qopt[iMin[0], jMin[0], kMin[0], a_]
    # Qopt[iMin, jMin, kMax]          
    with hcl.if_(hcl.and_(iMin[0] < Qopt.shape[0], jMin[0] < Qopt.shape[1], kMax[0] < Qopt.shape[2])):
        with hcl.if_(hcl.and_(iMin[0] >= 0, jMin[0] >= 0, kMax[0] >= 0)):
            with hcl.for_(0, actions.shape[0], name="a_") as a_:
                with hcl.if_(c001[0] < Qopt[iMin[0], jMin[0], kMax[0], a_]):
                    c001[0] = Qopt[iMin[0], jMin[0], kMax[0], a_]
    # Qopt[iMin, jMax, kMin]  
    with hcl.if_(hcl.and_(iMin[0] < Qopt.shape[0], jMax[0] < Qopt.shape[1], kMin[0] < Qopt.shape[2])):
        with hcl.if_(hcl.and_(iMin[0] >= 0, jMax[0] >= 0, kMin[0] >= 0)):
            with hcl.for_(0, actions.shape[0], name="a_") as a_:
                with hcl.if_(c010[0] < Qopt[iMin[0], jMax[0], kMin[0], a_]):
                    c010[0] = Qopt[iMin[0], jMax[0], kMin[0], a_]
    # Qopt[iMin, jMax, kMax]  
    with hcl.if_(hcl.and_(iMin[0] < Qopt.shape[0], jMax[0] < Qopt.shape[1], kMax[0] < Qopt.shape[2])):
        with hcl.if_(hcl.and_(iMin[0] >= 0, jMax[0] >= 0, kMax[0] >= 0)):
            with hcl.for_(0, actions.shape[0], name="a_") as a_:
                with hcl.if_(c011[0] < Qopt[iMin[0], jMax[0], kMax[0], a_]):
                    c011[0] = Qopt[iMin[0], jMax[0], kMax[0], a_]
    # Qopt[iMax, jMin, kMin]  
    with hcl.if_(hcl.and_(iMax[0] < Qopt.shape[0], jMin[0] < Qopt.shape[1], kMin[0] < Qopt.shape[2])):
        with hcl.if_(hcl.and_(iMax[0] >= 0, jMin[0] >= 0, kMin[0] >= 0)):
            with hcl.for_(0, actions.shape[0], name="a_") as a_:
                with hcl.if_(c100[0] < Qopt[iMax[0], jMin[0], kMin[0], a_]):
                    c100[0] = Qopt[iMax[0], jMin[0], kMin[0], a_]
    # Qopt[iMax, jMin, kMax] 
    with hcl.if_(hcl.and_(iMax[0] < Qopt.shape[0], jMin[0] < Qopt.shape[1], kMax[0] < Qopt.shape[2])):
        with hcl.if_(hcl.and_(iMax[0] >= 0, jMin[0] >= 0, kMax[0] >= 0)):
            with hcl.for_(0, actions.shape[0], name="a_") as a_:
                with hcl.if_(c101[0] < Qopt[iMax[0], jMin[0], kMax[0], a_]):
                    c101[0] = Qopt[iMax[0], jMin[0], kMax[0], a_]
    # Qopt[iMax, jMax, kMin]
    with hcl.if_(hcl.and_(iMax[0] < Qopt.shape[0], jMax[0] < Qopt.shape[1], kMin[0] < Qopt.shape[2])):
        with hcl.if_(hcl.and_(iMax[0] >= 0, jMax[0] >= 0, kMin[0] >= 0)):
            with hcl.for_(0, actions.shape[0], name="a_") as a_:
                with hcl.if_(c110[0] < Qopt[iMax[0], jMax[0], kMin[0], a_]):
                    c110[0] = Qopt[iMax[0], jMax[0], kMin[0], a_]
    # Qopt[iMax, jMax, kMax]
    with hcl.if_(hcl.and_(iMax[0] < Qopt.shape[0], jMax[0] < Qopt.shape[1], kMax[0] < Qopt.shape[2])):
        with hcl.if_(hcl.and_(iMax[0] >= 0, jMax[0] >= 0, kMax[0] >= 0)):
            with hcl.for_(0, actions.shape[0], name="a_") as a_:
                with hcl.if_(c111[0] < Qopt[iMax[0], jMax[0], kMax[0], a_]):
                    c111[0] = Qopt[iMax[0], jMax[0], kMax[0], a_]

    # perform linear interpolation
    c00[0] = (c000[0] * (1-di[0])) + (c100[0] * di[0])
    c01[0] = (c001[0] * (1-di[0])) + (c101[0] * di[0])
    c10[0] = (c010[0] * (1-di[0])) + (c110[0] * di[0])
    c11[0] = (c011[0] * (1-di[0])) + (c111[0] * di[0])
    c0[0]  = (c00[0] * (1-dj[0])) + (c10[0] * dj[0])
    c1[0]  = (c01[0] * (1-dj[0])) + (c11[0] * dj[0])
    interpV[0] = (c0[0] * (1-dk[0])) + (c1[0] * dk[0])



######################################### VALUE ITERATION ##########################################


def value_iteration_3D():
    def solve_Qopt(Qopt, actions, intermeds, trans, interpV, gamma, epsilon, iVals, sVals, bounds, goal, ptsEachDim, count, maxIters, useNN, fillVal):
            reSweep = hcl.scalar(1, "reSweep")
            oldQ    = hcl.scalar(0, "oldV")
            newQ    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < maxIters[0])):
                reSweep[0] = 0
                # Perform value iteration by sweeping in direction 1
                with hcl.Stage("Sweep_1"):
                    with hcl.for_(0, Qopt.shape[0], name="i") as i:
                        with hcl.for_(0, Qopt.shape[1], name="j") as j:
                            with hcl.for_(0, Qopt.shape[2], name="k") as k:
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i,j,k,a]
                                    updateQopt(i, j, k, a, iVals, sVals, Qopt, actions, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newQ[0] = Qopt[i,j,k,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 2
                with hcl.Stage("Sweep_2"):
                    # For all states
                    with hcl.for_(1, Qopt.shape[0] + 1, name="i") as i:
                        with hcl.for_(1, Qopt.shape[1] + 1, name="j") as j:
                            with hcl.for_(1, Qopt.shape[2] + 1, name="k") as k:
                                i2 = Qopt.shape[0] - i
                                j2 = Qopt.shape[1] - j
                                k2 = Qopt.shape[2] - k
                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i2,j2,k2,a]
                                    updateQopt(i2, j2, k2, a, iVals, sVals, Qopt, actions, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newQ[0] = Qopt[i2,j2,k2,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 3
                with hcl.Stage("Sweep_3"):
                    # For all states
                    with hcl.for_(1, Qopt.shape[0] + 1, name="i") as i:
                        with hcl.for_(0, Qopt.shape[1], name="j") as j:
                            with hcl.for_(0, Qopt.shape[2], name="k") as k:
                                i2 = Qopt.shape[0] - i
                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i2,j,k,a]
                                    updateQopt(i2, j, k, a, iVals, sVals, Qopt, actions, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newQ[0] = Qopt[i2,j,k,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 4
                with hcl.Stage("Sweep_4"):
                    # For all states
                    with hcl.for_(0, Qopt.shape[0], name="i") as i:
                        with hcl.for_(1, Qopt.shape[1] + 1, name="j") as j:
                            with hcl.for_(0, Qopt.shape[2], name="k") as k:
                                j2 = Qopt.shape[1] - j
                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i,j2,k,a]
                                    updateQopt(i, j2, k, a, iVals, sVals, Qopt, actions, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newQ[0] = Qopt[i,j2,k,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 5
                with hcl.Stage("Sweep_5"):
                    # For all states
                    with hcl.for_(0, Qopt.shape[0], name="i") as i:
                        with hcl.for_(0, Qopt.shape[1], name="j") as j:
                            with hcl.for_(1, Qopt.shape[2] + 1, name="k") as k:
                                k2 = Qopt.shape[2] - k
                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i,j,k2,a]
                                    updateQopt(i, j, k2, a, iVals, sVals, Qopt, actions, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newQ[0] = Qopt[i,j,k2,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 6
                with hcl.Stage("Sweep_6"):
                    # For all states
                    with hcl.for_(1, Qopt.shape[0] + 1, name="i") as i:
                        with hcl.for_(1, Qopt.shape[1] + 1, name="j") as j:
                            with hcl.for_(0, Qopt.shape[2], name="k") as k:
                                i2 = Qopt.shape[0] - i
                                j2 = Qopt.shape[1] - j
                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i2,j2,k,a]
                                    updateQopt(i2, j2, k, a, iVals, sVals, Qopt, actions, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newQ[0] = Qopt[i2,j2,k,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 7
                with hcl.Stage("Sweep_7"):
                    # For all states
                    with hcl.for_(1, Qopt.shape[0] + 1, name="i") as i:
                        with hcl.for_(0, Qopt.shape[1], name="j") as j:
                            with hcl.for_(1, Qopt.shape[2] + 1, name="k") as k:
                                i2 = Qopt.shape[0] - i
                                k2 = Qopt.shape[2] - k
                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i2,j,k2,a]
                                    updateQopt(i2, j, k2, a, iVals, sVals, Qopt, actions, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newQ[0] = Qopt[i2,j,k2,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 8
                with hcl.Stage("Sweep_8"):
                    # For all states
                    with hcl.for_(0, Qopt.shape[0], name="i") as i:
                        with hcl.for_(1, Qopt.shape[1] + 1, name="j") as j:
                            with hcl.for_(1, Qopt.shape[2] + 1, name="k") as k:
                                j2 = Qopt.shape[1] - j
                                k2 = Qopt.shape[2] - k
                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i,j2,k2,a]
                                    updateQopt(i, j2, k2, a, iVals, sVals, Qopt, actions, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newQ[0] = Qopt[i,j2,k2,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1


    ###################################### SETUP PLACEHOLDERS ######################################
    
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    # NOTE: trans is a tensor with size = maximum number of transitions
    # NOTE: intermeds must have size  [# possible actions]
    # NOTE: transition must have size [# possible outcomes, #state dimensions + 1]
    Qopt       = hcl.placeholder(tuple(UD._ptsEachDim), name="Qopt", dtype=hcl.Float())
    gamma      = hcl.placeholder((0,), "gamma")
    count      = hcl.placeholder((0,), "count")
    epsilon    = hcl.placeholder((0,), "epsilon")
    actions    = hcl.placeholder(tuple(UD._actions.shape), name="actions", dtype=hcl.Float())
    intermeds  = hcl.placeholder(tuple([UD._actions.shape[0]]), name="intermeds", dtype=hcl.Float())
    trans      = hcl.placeholder(tuple(UD._trans.shape), name="successors", dtype=hcl.Float())
    bounds     = hcl.placeholder(tuple(UD._bounds.shape), name="bounds", dtype=hcl.Float())
    goal       = hcl.placeholder(tuple(UD._goal.shape), name="goal", dtype=hcl.Float())
    ptsEachDim = hcl.placeholder(tuple([3]), name="ptsEachDim", dtype=hcl.Float())
    sVals      = hcl.placeholder(tuple([3]), name="sVals", dtype=hcl.Float())
    iVals      = hcl.placeholder(tuple([3]), name="iVals", dtype=hcl.Float())
    interpV    = hcl.placeholder((0,), "interpV")
    maxIters   = hcl.placeholder((0,), "maxIters")
    useNN      = hcl.placeholder((0,), "useNN")
    fillVal    = hcl.placeholder((0,), "fillVal")

    # Create a static schedule -- graph
    s = hcl.create_schedule([Qopt, actions, intermeds, trans, interpV, gamma, epsilon, iVals, sVals, bounds, goal, ptsEachDim, count, maxIters, useNN, fillVal], solve_Qopt)
    

    ########################################## INITIALIZE ##########################################

    # Convert the python array to hcl type array
    Q_opt      = hcl.asarray(np.zeros(UD._ptsEachDim))
    intermeds  = hcl.asarray(np.ones(UD._actions.shape[0]))
    trans      = hcl.asarray(UD._trans)
    gamma      = hcl.asarray(UD._gamma)
    epsilon    = hcl.asarray(UD._epsilon)
    count      = hcl.asarray(np.zeros(1))
    actions    = hcl.asarray(UD._actions)
    bounds     = hcl.asarray(UD._bounds)
    goal       = hcl.asarray(UD._goal)
    ptsEachDim = hcl.asarray(UD._ptsEachDim)
    sVals      = hcl.asarray(np.zeros([3]))
    iVals      = hcl.asarray(np.zeros([3]))
    interpV    = hcl.asarray(np.zeros([1])) 
    maxIters   = hcl.asarray(UD._maxIters)
    useNN      = hcl.asarray(UD._useNN)
    fillVal    = hcl.asarray(UD._fillVal)


    ########################################### EXECUTE ############################################

    # Use this graph and build an executable
    f = hcl.build(s, target="llvm")

    t_s = time.time()
    f(Q_opt, actions, intermeds, trans, interpV, gamma, epsilon, iVals, sVals, bounds, goal, ptsEachDim, count, maxIters, useNN, fillVal)
    t_e = time.time()

    Q = Q_opt.asnumpy()
    c = count.asnumpy()

    print()
    print("Finished in ", int(c[0]), " iterations")
    print("Took        ", t_e-t_s, " seconds")

    dir_path  = "./hcl_value_matrix_test/"
    file_name = "hcl_value_iteration_Qvalue_" + str(int(c[0])) + "_iterations_by" + ("_Interpolation" if UD._useNN[0] == 0 else "_NN")
    UD.writeResults(Q, dir_path, file_name, just_values=True)


# Test function
value_iteration_3D()
