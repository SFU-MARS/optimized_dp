import heterocl as hcl
import numpy as np

######################################### HELPER FUNCTIONS #########################################


# Update the value function at position (i,j,k,l,m,n)
# iVals:      holds index values (i,j,k,l,m,n) that correspond to state values (si,sj,sk,sl,sm,sn)
# intermeds:  holds the estimated value associated with taking each action
# interpV:    holds the estimated value of a successor state (linear interpolation only)
# gamma:      discount factor
# ptsEachDim: the number of grid points in each dimension of the state space
# useNN:      a mode flag (0: use linear interpolation, 1: use nearest neighbour)
def updateVopt(obj, i, j, k, l, m, n, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN):
    p = hcl.scalar(0, "p")

    with hcl.for_(0, actions.shape[0], name="a") as a:
        # set iVals equal to (i,j,k,l,m,n) and sVals equal to the corresponding state values (si,sj,sk,sl,sm,sn)
        updateStateVals(i, j, k, l, m, n, iVals, sVals, bounds, ptsEachDim)
        # call the transition function to obtain the outcome(s) of action a from state (si,sj,sk,sl,sm,sn)
        obj.transition(sVals, actions[a], bounds, trans, goal)
        # initialize the value of the action Q value with the immediate reward of taking that action
        intermeds[a] = obj.reward(sVals, actions[a], bounds, goal, trans)
        # add the value of each possible successor state to the Q value
        with hcl.for_(0, trans.shape[0], name="si") as si:
            p[0]     = trans[si,0]
            sVals[0] = trans[si,1]
            sVals[1] = trans[si,2]
            sVals[2] = trans[si,3]
            sVals[3] = trans[si,4]
            sVals[4] = trans[si,5]
            sVals[5] = trans[si,6]

            # Nearest neighbour
            with hcl.if_(useNN[0] == 1):
                # convert the state values of the successor state (si,sj,sk,sl,sm,sn) into indeces (ia,ja,ka,la,ma,na)
                stateToIndex(sVals, iVals, bounds, ptsEachDim)
                # if (ia,ja,ka,la,ma,na) is within the state space, add its discounted value to the Q value
                with hcl.if_(hcl.and_(iVals[0] < Vopt.shape[0], iVals[1] < Vopt.shape[1], iVals[2] < Vopt.shape[2])):
                    with hcl.if_(hcl.and_(iVals[3] < Vopt.shape[3], iVals[4] < Vopt.shape[4], iVals[5] < Vopt.shape[5])):
                        with hcl.if_(hcl.and_(iVals[0] >= 0, iVals[1] >= 0, iVals[2] >= 0, iVals[3] >= 0, iVals[4] >= 0, iVals[5] >= 0)):
                            intermeds[a] += (gamma[0] * (p[0] * Vopt[iVals[0], iVals[1], iVals[2], iVals[3], iVals[4], iVals[5]]))

        # maximize over each Q value to obtain the optimal value
        Vopt[i,j,k,l,m,n] = -1000000
        with hcl.for_(0, intermeds.shape[0], name="r") as r:
            with hcl.if_(Vopt[i,j,k,l,m,n] < intermeds[r]):
                Vopt[i,j,k,l,m,n] = intermeds[r]


# Returns 0 if convergence has been reached
def evaluateConvergence(newV, oldV, epsilon, reSweep):
    delta = hcl.scalar(0, "delta")
    # Calculate the difference, if it's negative, make it positive
    delta[0] = newV[0] - oldV[0]
    with hcl.if_(delta[0] < 0):
        delta[0] = delta[0] * -1
    with hcl.if_(delta[0] > epsilon[0]):
        reSweep[0] = 1


# Converts state values into indeces using nearest neighbour rounding
def stateToIndex(sVals, iVals, bounds, ptsEachDim):
    iVals[0] = ((sVals[0] - bounds[0,0]) / (bounds[0,1] - bounds[0,0])) *  (ptsEachDim[0] - 1)
    iVals[1] = ((sVals[1] - bounds[1,0]) / (bounds[1,1] - bounds[1,0])) *  (ptsEachDim[1] - 1)
    iVals[2] = ((sVals[2] - bounds[2,0]) / (bounds[2,1] - bounds[2,0])) *  (ptsEachDim[2] - 1)
    iVals[3] = ((sVals[3] - bounds[3,0]) / (bounds[3,1] - bounds[3,0])) *  (ptsEachDim[3] - 1)
    iVals[4] = ((sVals[4] - bounds[4,0]) / (bounds[4,1] - bounds[4,0])) *  (ptsEachDim[4] - 1)
    iVals[5] = ((sVals[5] - bounds[5,0]) / (bounds[5,1] - bounds[5,0])) *  (ptsEachDim[5] - 1)
    # NOTE: add 0.5 to simulate rounding
    iVals[0] = hcl.cast(hcl.Int(), iVals[0] + 0.5)
    iVals[1] = hcl.cast(hcl.Int(), iVals[1] + 0.5)
    iVals[2] = hcl.cast(hcl.Int(), iVals[2] + 0.5)
    iVals[3] = hcl.cast(hcl.Int(), iVals[3] + 0.5)
    iVals[4] = hcl.cast(hcl.Int(), iVals[4] + 0.5)
    iVals[5] = hcl.cast(hcl.Int(), iVals[5] + 0.5)


# Convert indices into state values
def indexToState(iVals, sVals, bounds, ptsEachDim):
    sVals[0] = bounds[0,0] + ( (bounds[0,1] - bounds[0,0]) * (iVals[0] / (ptsEachDim[0]-1)) ) 
    sVals[1] = bounds[1,0] + ( (bounds[1,1] - bounds[1,0]) * (iVals[1] / (ptsEachDim[1]-1)) ) 
    sVals[2] = bounds[2,0] + ( (bounds[2,1] - bounds[2,0]) * (iVals[2] / (ptsEachDim[2]-1)) ) 
    sVals[3] = bounds[3,0] + ( (bounds[3,1] - bounds[3,0]) * (iVals[3] / (ptsEachDim[3]-1)) ) 
    sVals[4] = bounds[4,0] + ( (bounds[4,1] - bounds[4,0]) * (iVals[4] / (ptsEachDim[4]-1)) ) 
    sVals[5] = bounds[5,0] + ( (bounds[5,1] - bounds[5,0]) * (iVals[5] / (ptsEachDim[5]-1)) ) 


# Sets iVals equal to (i,j,k,l,m,n) and sVals equal to the corresponding state values
def updateStateVals(i, j, k, l, m, n, iVals, sVals, bounds, ptsEachDim):
    iVals[0] = i
    iVals[1] = j
    iVals[2] = k
    iVals[3] = l
    iVals[4] = m
    iVals[5] = n
    indexToState(iVals, sVals, bounds, ptsEachDim)



######################################### VALUE ITERATION ##########################################


# Main value iteration algorithm
# reSweep:  a convergence flag (1: continue iterating, 0: convergence reached)
# epsilon:  convergence criteria
# maxIters: maximum number of iterations that can occur without convergence being reached
# count:    the number of iterations that have been performed
def value_iteration_6D(MDP_obj):
    def solve_Vopt(Vopt, actions, intermeds, trans, interpV, gamma, epsilon, iVals, sVals, bounds, goal, ptsEachDim, count, maxIters, useNN):
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
                                with hcl.for_(0, Vopt.shape[3], name="l") as l:
                                    with hcl.for_(0, Vopt.shape[4], name="m") as m:
                                        with hcl.for_(0, Vopt.shape[5], name="n") as n:
                                            oldV[0] = Vopt[i,j,k,l,m,n]
                                            updateVopt(MDP_obj, i, j, k, l, m, n, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN)
                                            newV[0] = Vopt[i,j,k,l,m,n]
                                            evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 2
                with hcl.Stage("Sweep_2"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                            with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                                with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                    with hcl.for_(0, Vopt.shape[3], name="l") as l:
                                        with hcl.for_(0, Vopt.shape[4], name="m") as m:
                                            with hcl.for_(0, Vopt.shape[5], name="n") as n:
                                                i2 = Vopt.shape[0] - i
                                                j2 = Vopt.shape[1] - j
                                                k2 = Vopt.shape[2] - k
                                                oldV[0] = Vopt[i2,j2,k2,l,m,n]
                                                updateVopt(MDP_obj, i2, j2, k2, l, m, n, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN)
                                                newV[0] = Vopt[i2,j2,k2,l,m,n]
                                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 3
                with hcl.Stage("Sweep_3"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                            with hcl.for_(0, Vopt.shape[1], name="j") as j:
                                with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                    with hcl.for_(0, Vopt.shape[3], name="l") as l:
                                        with hcl.for_(0, Vopt.shape[4], name="m") as m:
                                            with hcl.for_(0, Vopt.shape[5], name="n") as n:
                                                i2 = Vopt.shape[0] - i
                                                oldV[0] = Vopt[i2,j,k,l,m,n]
                                                updateVopt(MDP_obj, i2, j, k, l, m, n, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN)
                                                newV[0] = Vopt[i2,j,k,l,m,n]
                                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 4
                with hcl.Stage("Sweep_4"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(0, Vopt.shape[0], name="i") as i:
                            with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                                with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                    with hcl.for_(0, Vopt.shape[3], name="l") as l:
                                        with hcl.for_(0, Vopt.shape[4], name="m") as m:
                                            with hcl.for_(0, Vopt.shape[5], name="n") as n:
                                                j2 = Vopt.shape[1] - j
                                                oldV[0] = Vopt[i,j2,k,l,m,n]
                                                updateVopt(MDP_obj, i, j2, k, l, m, n, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN)
                                                newV[0] = Vopt[i,j2,k,l,m,n]
                                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 5
                with hcl.Stage("Sweep_5"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(0, Vopt.shape[0], name="i") as i:
                            with hcl.for_(0, Vopt.shape[1], name="j") as j:
                                with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                    with hcl.for_(0, Vopt.shape[3], name="l") as l:
                                        with hcl.for_(0, Vopt.shape[4], name="m") as m:
                                            with hcl.for_(0, Vopt.shape[5], name="n") as n:
                                                k2 = Vopt.shape[2] - k
                                                oldV[0] = Vopt[i,j,k2,l,m,n]
                                                updateVopt(MDP_obj, i, j, k2, l, m, n, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN)
                                                newV[0] = Vopt[i,j,k2,l,m,n]
                                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 6
                with hcl.Stage("Sweep_6"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                            with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                                with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                    with hcl.for_(0, Vopt.shape[3], name="l") as l:
                                        with hcl.for_(0, Vopt.shape[4], name="m") as m:
                                            with hcl.for_(0, Vopt.shape[5], name="n") as n:
                                                i2 = Vopt.shape[0] - i
                                                j2 = Vopt.shape[1] - j
                                                oldV[0] = Vopt[i2,j2,k,l,m,n]
                                                updateVopt(MDP_obj, i2, j2, k, l, m, n, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN)
                                                newV[0] = Vopt[i2,j2,k,l,m,n]
                                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 7
                with hcl.Stage("Sweep_7"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                            with hcl.for_(0, Vopt.shape[1], name="j") as j:
                                with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                    with hcl.for_(0, Vopt.shape[3], name="l") as l:
                                        with hcl.for_(0, Vopt.shape[4], name="m") as m:
                                            with hcl.for_(0, Vopt.shape[5], name="n") as n:
                                                i2 = Vopt.shape[0] - i
                                                k2 = Vopt.shape[2] - k
                                                oldV[0] = Vopt[i2,j,k2,l,m,n]
                                                updateVopt(MDP_obj, i2, j, k2, l, m, n, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN)
                                                newV[0] = Vopt[i2,j,k2,l,m,n]
                                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1
                # Perform value iteration by sweeping in direction 8
                with hcl.Stage("Sweep_8"):
                    with hcl.if_(useNN[0] == 1):
                        with hcl.for_(0, Vopt.shape[0], name="i") as i:
                            with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                                with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                    with hcl.for_(0, Vopt.shape[3], name="l") as l:
                                        with hcl.for_(0, Vopt.shape[4], name="m") as m:
                                            with hcl.for_(0, Vopt.shape[5], name="n") as n:
                                                j2 = Vopt.shape[1] - j
                                                k2 = Vopt.shape[2] - k
                                                oldV[0] = Vopt[i,j2,k2,l,m,n]
                                                updateVopt(MDP_obj, i, j2, k2, l, m, n, iVals, sVals, actions, Vopt, intermeds, trans, interpV, gamma, bounds, goal, ptsEachDim, useNN)
                                                newV[0] = Vopt[i,j2,k2,l,m,n]
                                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                        count[0] += 1


    ###################################### SETUP PLACEHOLDERS ######################################
    
    # Initialize the HCL environment
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    # NOTE: trans is a tensor with size = maximum number of transitions
    # NOTE: intermeds must have size  [# possible actions]
    # NOTE: transition must have size [# possible outcomes, #state dimensions + 1]
    Vopt       = hcl.placeholder(tuple(MDP_obj._ptsEachDim), name="Vopt", dtype=hcl.Float())
    gamma      = hcl.placeholder((0,), "gamma")
    count      = hcl.placeholder((0,), "count")
    maxIters   = hcl.placeholder((0,), "maxIters")
    epsilon    = hcl.placeholder((0,), "epsilon")
    actions    = hcl.placeholder(tuple(MDP_obj._actions.shape), name="actions", dtype=hcl.Float())
    intermeds  = hcl.placeholder(tuple([MDP_obj._actions.shape[0]]), name="intermeds", dtype=hcl.Float())
    trans      = hcl.placeholder(tuple(MDP_obj._trans.shape), name="successors", dtype=hcl.Float())
    bounds     = hcl.placeholder(tuple(MDP_obj._bounds.shape), name="bounds", dtype=hcl.Float())
    goal       = hcl.placeholder(tuple(MDP_obj._goal.shape), name="goal", dtype=hcl.Float())
    ptsEachDim = hcl.placeholder(tuple([6]), name="ptsEachDim", dtype=hcl.Float())
    sVals      = hcl.placeholder(tuple([6]), name="sVals", dtype=hcl.Float())
    iVals      = hcl.placeholder(tuple([6]), name="iVals", dtype=hcl.Float())
    interpV    = hcl.placeholder((0,), "interpols")
    useNN      = hcl.placeholder((0,), "useNN")

    # Create a static schedule -- graph
    s = hcl.create_schedule([Vopt, actions, intermeds, trans, interpV, gamma, epsilon, iVals, sVals, bounds, goal, ptsEachDim, count, maxIters, useNN], solve_Vopt)

    # Use this graph and build an executable
    return hcl.build(s, target="llvm")
