import heterocl as hcl
import numpy as np

######################################### HELPER FUNCTIONS #########################################


# Update the value function at position (i,j,k)
# iVals:      holds index values (i,j,k) that correspond to state values (si,sj,sk)
# intermeds:  holds the estimated value associated with taking each action
# interpV:    holds the estimated value of a successor state (linear interpolation only)
# gamma:      discount factor
# ptsEachDim: the number of grid points in each dimension of the state space
# useNN:      a mode flag (0: use linear interpolation, 1: use nearest neighbour)
def updateVopt(obj, i, j, k, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal):
    p = hcl.scalar(0, "p")

    with hcl.for_(0, actions.shape[0], name="a") as a:
        # set iVals equal to (i,j,k) and sVals equal to the corresponding state values (si,sj,sk)
        updateStateVals(i, j, k, iVals, sVals, bounds, ptsEachDim)
        # call the transition function to obtain the outcome(s) of action a from state (si,sj,sk)
        obj.transition(sVals, actions[a], bounds, trans, goal)
        # initialize the value of the action using the immediate reward of taking that action
        intermeds[a] = obj.reward(sVals, actions[a], bounds, goal, trans)
        Vopt[i,j,k]  = intermeds[a] 
        # add the value of each possible successor state to the estimated value of taking action a
        with hcl.for_(0, trans.shape[0], name="si") as si:
            p[0]     = trans[si,0]
            sVals[0] = trans[si,1]
            sVals[1] = trans[si,2]
            sVals[2] = trans[si,3]
            # Nearest neighbour
            with hcl.if_(useNN[0] == 1):
                # convert the state values of the successor state (si,sj,sk) into indeces (ia,ij,ik)
                stateToIndex(sVals, iVals, bounds, ptsEachDim)
                # if (ia, ij, ik) is within the state space, add its discounted value to action a
                with hcl.if_(hcl.and_(iVals[0] < Vopt.shape[0], iVals[1] < Vopt.shape[1], iVals[2] < Vopt.shape[2])):
                    with hcl.if_(hcl.and_(iVals[0] >= 0, iVals[1] >= 0, iVals[2] >= 0)):
                        intermeds[a] += (gamma[0] * (p[0] * Vopt[iVals[0], iVals[1], iVals[2]]))
            # Linear interpolation
            with hcl.if_(useNN[0] == 0):
                # if (sia, sja, ska) is within the state space, add its discounted value to action a
                with hcl.if_(hcl.and_(sVals[0] <= bounds[0,1], sVals[1] <= bounds[1,1], sVals[2] <= bounds[2,1])):
                    with hcl.if_(hcl.and_(sVals[0] >= bounds[0,0], sVals[1] >= bounds[1,0], sVals[2] >= bounds[2,0])):
                        stateToIndexInterpolants(Vopt, sVals, bounds, ptsEachDim, interpV, fillVal)
                        intermeds[a] += (gamma[0] * (p[0] * interpV[0]))
        # maximize over each possible action in intermeds to obtain the optimal value
        with hcl.for_(0, intermeds.shape[0], name="r") as r:
            with hcl.if_(Vopt[i,j,k] < intermeds[r]):
                Vopt[i,j,k] = intermeds[r]


# Returns 0 if convergence has been reached
def evaluateConvergence(newV, oldV, epsilon, reSweep):
    delta = hcl.scalar(0, "delta")
    # Calculate the difference, if it's negative, make it positive
    delta[0] = newV[0] - oldV[0]
    with hcl.if_(delta[0] < 0):
        delta[0] = delta[0] * -1
    with hcl.if_(delta[0] > epsilon[0]):
        reSweep[0] = 1


# _bounds = np.array([[-5.0, 5.0],[-5.0, 5.0],[-3.1415, 3.1415]])
# convert state values into indeces using nearest neighbour
def stateToIndex(sVals, iVals, bounds, ptsEachDim):
    iVals[0] = ((sVals[0] - bounds[0,0]) / (bounds[0,1] - bounds[0,0])) *  (ptsEachDim[0] - 1)
    iVals[1] = ((sVals[1] - bounds[1,0]) / (bounds[1,1] - bounds[1,0])) *  (ptsEachDim[1] - 1)
    iVals[2] = ((sVals[2] - bounds[2,0]) / (bounds[2,1] - bounds[2,0])) *  (ptsEachDim[2] - 1)
    # NOTE: add 0.5 to simulate rounding
    iVals[0] = hcl.cast(hcl.Int(), iVals[0] + 0.5)
    iVals[1] = hcl.cast(hcl.Int(), iVals[1] + 0.5)
    iVals[2] = hcl.cast(hcl.Int(), iVals[2] + 0.5)


# given state values sVals, obtain the 8 possible successor states and their corresponding weight
def stateToIndexInterpolants(Vopt, sVals, bounds, ptsEachDim, interpV, fillVal):
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
    # Vopt[iMin, jMin, kMin]
    with hcl.if_(hcl.and_(iMin[0] < Vopt.shape[0], jMin[0] < Vopt.shape[1], kMin[0] < Vopt.shape[2])):
        with hcl.if_(hcl.and_(iMin[0] >= 0, jMin[0] >= 0, kMin[0] >= 0)):
            c000[0] = Vopt[iMin[0], jMin[0], kMin[0]]
    # Vopt[iMin, jMin, kMax]
    with hcl.if_(hcl.and_(iMin[0] < Vopt.shape[0], jMin[0] < Vopt.shape[1], kMax[0] < Vopt.shape[2])):
        with hcl.if_(hcl.and_(iMin[0] >= 0, jMin[0] >= 0, kMax[0] >= 0)):
            c001[0] = Vopt[iMin[0], jMin[0], kMax[0]]
    # Vopt[iMin, jMax, kMin] 
    with hcl.if_(hcl.and_(iMin[0] < Vopt.shape[0], jMax[0] < Vopt.shape[1], kMin[0] < Vopt.shape[2])):
        with hcl.if_(hcl.and_(iMin[0] >= 0, jMax[0] >= 0, kMin[0] >= 0)):
            c010[0] = Vopt[iMin[0], jMax[0], kMin[0]]
    # Vopt[iMin, jMax, kMax] 
    with hcl.if_(hcl.and_(iMin[0] < Vopt.shape[0], jMax[0] < Vopt.shape[1], kMax[0] < Vopt.shape[2])):
        with hcl.if_(hcl.and_(iMin[0] >= 0, jMax[0] >= 0, kMax[0] >= 0)):
            c011[0] = Vopt[iMin[0], jMax[0], kMax[0]]
    # Vopt[iMax, jMin, kMin] 
    with hcl.if_(hcl.and_(iMax[0] < Vopt.shape[0], jMin[0] < Vopt.shape[1], kMin[0] < Vopt.shape[2])):
        with hcl.if_(hcl.and_(iMax[0] >= 0, jMin[0] >= 0, kMin[0] >= 0)):
            c100[0] = Vopt[iMax[0], jMin[0], kMin[0]]
    # Vopt[iMax, jMin, kMax]
    with hcl.if_(hcl.and_(iMax[0] < Vopt.shape[0], jMin[0] < Vopt.shape[1], kMax[0] < Vopt.shape[2])):
        with hcl.if_(hcl.and_(iMax[0] >= 0, jMin[0] >= 0, kMax[0] >= 0)):
            c101[0] = Vopt[iMax[0], jMin[0], kMax[0]]
    # Vopt[iMax, jMax, kMin]
    with hcl.if_(hcl.and_(iMax[0] < Vopt.shape[0], jMax[0] < Vopt.shape[1], kMin[0] < Vopt.shape[2])):
        with hcl.if_(hcl.and_(iMax[0] >= 0, jMax[0] >= 0, kMin[0] >= 0)):
            c110[0] = Vopt[iMax[0], jMax[0], kMin[0]]
    # Vopt[iMax, jMax, kMax]
    with hcl.if_(hcl.and_(iMax[0] < Vopt.shape[0], jMax[0] < Vopt.shape[1], kMax[0] < Vopt.shape[2])):
        with hcl.if_(hcl.and_(iMax[0] >= 0, jMax[0] >= 0, kMax[0] >= 0)):
            c111[0] = Vopt[iMax[0], jMax[0], kMax[0]]

    # perform linear interpolation
    c00[0] = (c000[0] * (1-di[0])) + (c100[0] * di[0])
    c01[0] = (c001[0] * (1-di[0])) + (c101[0] * di[0])
    c10[0] = (c010[0] * (1-di[0])) + (c110[0] * di[0])
    c11[0] = (c011[0] * (1-di[0])) + (c111[0] * di[0])
    c0[0]  = (c00[0] * (1-dj[0])) + (c10[0] * dj[0])
    c1[0]  = (c01[0] * (1-dj[0])) + (c11[0] * dj[0])
    interpV[0] = (c0[0] * (1-dk[0])) + (c1[0] * dk[0])
    

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



######################################### VALUE ITERATION ##########################################


# Main value iteration algorithm
# reSweep:  a convergence flag (1: continue iterating, 0: convergence reached)
# epsilon:  convergence criteria
# maxIters: maximum number of iterations that can occur without convergence being reached
# count:    the number of iterations that have been performed
def value_iteration_3D(MDP_object):
    def solve_Vopt(Vopt, actions, intermeds, trans, interpV, gamma, epsilon, iVals, sVals, bounds, goal, ptsEachDim, count, maxIters, useNN, fillVal):
            reSweep = hcl.scalar(1, "reSweep")
            oldV    = hcl.scalar(0, "oldV")
            newV    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < maxIters[0])):
                reSweep[0] = 0
                # Perform value iteration by sweeping in direction 1
                with hcl.Stage("Sweep_1"):
                    with hcl.for_(0, Vopt.shape[0], name="i") as i:
                        with hcl.for_(0, Vopt.shape[1], name="j") as j:
                            with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                oldV[0] = Vopt[i,j,k]
                                updateVopt(MDP_object, i, j, k, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                newV[0] = Vopt[i,j,k]
                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 2
                with hcl.Stage("Sweep_2"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                            with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                                with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                    i2 = Vopt.shape[0] - i
                                    j2 = Vopt.shape[1] - j
                                    k2 = Vopt.shape[2] - k
                                    oldV[0] = Vopt[i2,j2,k2]
                                    updateVopt(MDP_object, i2, j2, k2, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newV[0] = Vopt[i2,j2,k2]
                                    evaluateConvergence(newV, oldV, epsilon, reSweep)
                        count[0] += 1
                # Perform value iteration by sweeping in direction 3
                with hcl.Stage("Sweep_3"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                            with hcl.for_(0, Vopt.shape[1], name="j") as j:
                                with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                    i2 = Vopt.shape[0] - i
                                    oldV[0] = Vopt[i2,j,k]
                                    updateVopt(MDP_object, i2, j, k, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newV[0] = Vopt[i2,j,k]
                                    evaluateConvergence(newV, oldV, epsilon, reSweep)
                        count[0] += 1
                # Perform value iteration by sweeping in direction 4
                with hcl.Stage("Sweep_4"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(0, Vopt.shape[0], name="i") as i:
                            with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                                with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                    j2 = Vopt.shape[1] - j
                                    oldV[0] = Vopt[i,j2,k]
                                    updateVopt(MDP_object, i, j2, k, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newV[0] = Vopt[i,j2,k]
                                    evaluateConvergence(newV, oldV, epsilon, reSweep)
                        count[0] += 1
                # Perform value iteration by sweeping in direction 5
                with hcl.Stage("Sweep_5"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(0, Vopt.shape[0], name="i") as i:
                            with hcl.for_(0, Vopt.shape[1], name="j") as j:
                                with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                    k2 = Vopt.shape[2] - k
                                    oldV[0] = Vopt[i,j,k2]
                                    updateVopt(MDP_object, i, j, k2, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newV[0] = Vopt[i,j,k2]
                                    evaluateConvergence(newV, oldV, epsilon, reSweep)
                        count[0] += 1
                # Perform value iteration by sweeping in direction 6
                with hcl.Stage("Sweep_6"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                            with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                                with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                    i2 = Vopt.shape[0] - i
                                    j2 = Vopt.shape[1] - j
                                    oldV[0] = Vopt[i2,j2,k]
                                    updateVopt(MDP_object, i2, j2, k, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newV[0] = Vopt[i2,j2,k]
                                    evaluateConvergence(newV, oldV, epsilon, reSweep)
                        count[0] += 1
                # Perform value iteration by sweeping in direction 7
                with hcl.Stage("Sweep_7"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                            with hcl.for_(0, Vopt.shape[1], name="j") as j:
                                with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                    i2 = Vopt.shape[0] - i
                                    k2 = Vopt.shape[2] - k
                                    oldV[0] = Vopt[i2,j,k2]
                                    updateVopt(MDP_object, i2, j, k2, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newV[0] = Vopt[i2,j,k2]
                                    evaluateConvergence(newV, oldV, epsilon, reSweep)
                        count[0] += 1
                # Perform value iteration by sweeping in direction 8
                with hcl.Stage("Sweep_8"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(0, Vopt.shape[0], name="i") as i:
                            with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                                with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                    j2 = Vopt.shape[1] - j
                                    k2 = Vopt.shape[2] - k
                                    oldV[0] = Vopt[i,j2,k2]
                                    updateVopt(MDP_object, i, j2, k2, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN, fillVal)
                                    newV[0] = Vopt[i,j2,k2]
                                    evaluateConvergence(newV, oldV, epsilon, reSweep)
                        count[0] += 1


    ###################################### SETUP PLACEHOLDERS ######################################
    

    # NOTE: trans is a tensor with size = maximum number of transitions
    # NOTE: intermeds must have size  [# possible actions]
    # NOTE: transition must have size [# possible outcomes, #state dimensions + 1]
    Vopt       = hcl.placeholder(tuple(MDP_object._ptsEachDim), name="Vopt", dtype=hcl.Float())
    gamma      = hcl.placeholder((0,), "gamma")
    count      = hcl.placeholder((0,), "count")
    maxIters   = hcl.placeholder((0,), "maxIters")
    epsilon    = hcl.placeholder((0,), "epsilon")
    actions    = hcl.placeholder(tuple(MDP_object._actions.shape), name="actions", dtype=hcl.Float())
    intermeds  = hcl.placeholder(tuple([MDP_object._actions.shape[0]]), name="intermeds", dtype=hcl.Float())
    trans      = hcl.placeholder(tuple(MDP_object._trans.shape), name="successors", dtype=hcl.Float())
    bounds     = hcl.placeholder(tuple(MDP_object._bounds.shape), name="bounds", dtype=hcl.Float())
    goal       = hcl.placeholder(tuple(MDP_object._goal.shape), name="goal", dtype=hcl.Float())
    ptsEachDim = hcl.placeholder(tuple([3]), name="ptsEachDim", dtype=hcl.Float())
    sVals      = hcl.placeholder(tuple([3]), name="sVals", dtype=hcl.Float())
    iVals      = hcl.placeholder(tuple([3]), name="iVals", dtype=hcl.Float())
    interpV    = hcl.placeholder((0,), "interpV")
    useNN      = hcl.placeholder((0,), "useNN")
    fillVal    = hcl.placeholder((0,), "fillVal")

    # Create a static schedule -- graph
    s = hcl.create_schedule([Vopt, actions, intermeds, trans, interpV, gamma, epsilon, iVals, sVals, bounds, goal, ptsEachDim, count, maxIters, useNN, fillVal], solve_Vopt)

    # Use this graph and build an executable
    return hcl.build(s) #target="llvm")