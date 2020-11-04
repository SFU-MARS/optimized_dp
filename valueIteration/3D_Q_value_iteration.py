import heterocl as hcl
#from Grid.GridProcessing import Grid
import numpy as np
import time
#from user_definer import *


####################################### USER-DEFINED VALUES ########################################


_bounds     = np.array([[-5.0, 5.0],[-3.0, 3.0],[10, 15]])
_goals      = np.array([[[-1.0, 1.0], [-1.0,1.0], [10,15]]]) # size: [number of goals x 3 x 2]
_ptsEachDim = np.array([40, 40, 50])
_actions    = np.array([[1,0,0], [0,1,0], [0,0,1]])
_gamma      = np.array([0.9])
_epsilon    = np.array([.0000005])
_maxIters   = np.array([500])
_trans      = np.zeros([2, 4]) # size: [maximum number of transition states available x 4]



###################################### USER-DEFINED FUNCTIONS ######################################


# Given state and action, return successor states and their probabilities
# sVals: the coordinates of state
def transition(sVals, action, trans):
    trans[0, 0] = 0.9
    trans[1, 0] = 0.1
    trans[0, 1] = sVals[0] + action[0]
    trans[0, 2] = sVals[1] + action[1]
    trans[0, 3] = sVals[2] + action[2]
    trans[1, 1] = sVals[0]
    trans[1, 2] = sVals[1]
    trans[1, 3] = sVals[2]


# Return the reward for taking action from state
def reward(sVals, action, trans, bounds, goals):
    rwd = hcl.scalar(0, "rwd")
    # for each possible successor
    with hcl.for_(0, trans.shape[0], name="si") as si:
        p        = trans[si,0]
        sVals[0] = trans[si,1]
        sVals[1] = trans[si,2]
        sVals[2] = trans[si,3]
        # Check if state is valid
        with hcl.if_(isStateValid(sVals, bounds)):
            # add reward for actions that take us to the goal
            with hcl.if_(isGoalState(sVals, goals) == 1):
                rwd[0] += (p * 100)
            # add default reward
            with hcl.else_():
                rwd[0] += p
    return rwd[0]




######################################### HELPER FUNCTIONS #########################################


# Update the value function at position (i,j,k)
def updateQopt(i, j, k, a, iVals, sVals, Qopt, actions, trans, gamma, bounds, goals, ptsEachDim):
    # set iVals equal to (i,j,k) and sVals equal to the corresponding state values at (i,j,k)
    updateStateVals(i, j, k, iVals, sVals, bounds, ptsEachDim)
    # call the transition function to obtain the outcome(s) of action a from state (si,sj,sk)
    transition(sVals, actions[a], trans)
    # initialize Qopt[i,j,k,a] with the immediate reward
    r             = hcl.scalar(0, "r")
    r[0]          = reward(sVals, actions[a], trans, bounds, goals)
    Qopt[i,j,k,a] = r[0]
    # maximize over successor Q-values
    with hcl.for_(0, trans.shape[0], name="si") as si:
        p        = trans[si,0]
        sVals[0] = trans[si,1]
        sVals[1] = trans[si,2]
        sVals[2] = trans[si,3]
        stateToIndex(sVals, iVals, bounds, ptsEachDim)
        # check if successor is within the grid
        with hcl.if_(hcl.and_(iVals[0] >= 0, iVals[0] < Qopt.shape[0])):
            with hcl.if_(hcl.and_(iVals[1] >= 0, iVals[1] < Qopt.shape[1])):
                with hcl.if_(hcl.and_(iVals[2] >= 0, iVals[2] < Qopt.shape[2])):
                    with hcl.for_(0, actions.shape[0], name="a_") as a_:
                        with hcl.if_(r + (gamma[0] * (p * Qopt[iVals[0],iVals[1],iVals[2],a_])) > Qopt[i,j,k,a]):
                            Qopt[i,j,k,a] = r + (gamma[0] * (p * Qopt[iVals[0],iVals[1],iVals[2],a_]))
        

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
    iVals[0] = (sVals[0] - bounds[0,0]) / ( (bounds[0,1] - bounds[0,0]) / (ptsEachDim[0]) )
    iVals[1] = (sVals[1] - bounds[1,0]) / ( (bounds[1,1] - bounds[1,0]) / (ptsEachDim[1]) )
    iVals[2] = (sVals[2] - bounds[2,0]) / ( (bounds[2,1] - bounds[2,0]) / (ptsEachDim[2]) )
    # NOTE: add 0.5 to simulate rounding
    iVals[0] = hcl.cast(hcl.Int(), iVals[0] + 0.5)
    iVals[1] = hcl.cast(hcl.Int(), iVals[1] + 0.5)
    iVals[2] = hcl.cast(hcl.Int(), iVals[2] + 0.5)


# convert indices into state values
def indexToState(iVals, sVals, bounds, ptsEachDim): 
    sVals[0] = (iVals[0] / ptsEachDim[0]) * (bounds[0,1] - bounds[0,0]) + bounds[0,0]
    sVals[1] = (iVals[1] / ptsEachDim[1]) * (bounds[1,1] - bounds[1,0]) + bounds[1,0]
    sVals[2] = (iVals[2] / ptsEachDim[2]) * (bounds[2,1] - bounds[2,0]) + bounds[2,0]


# set iVals equal to (i,j,k) and sVals equal to the corresponding state values at (i,j,k)
def updateStateVals(i, j, k, iVals, sVals, bounds, ptsEachDim):
    iVals[0] = i
    iVals[1] = j
    iVals[2] = k
    indexToState(iVals, sVals, bounds, ptsEachDim)


# check if state is in goal region
def isGoalState(sVals, goals):
    with hcl.for_(0, goals.shape[0], name="g") as g:
        # Check if state is valid
        with hcl.if_(hcl.and_(sVals[0] >= goals[g,0,0], sVals[0] <= goals[g,0,1])):
            with hcl.if_(hcl.and_(sVals[1] >= goals[g,1,0], sVals[0] <= goals[g,1,1])):
                with hcl.if_(hcl.and_(sVals[2] >= goals[g,2,0], sVals[0] <= goals[g,2,1])):
                    return 1
    return 0


# check if the sVals are within bounds
def isStateValid(sVals, bounds):
    with hcl.if_(hcl.and_(sVals[0] >= bounds[0,0], sVals[0] <= bounds[0,1])):
        with hcl.if_(hcl.and_(sVals[1] >= bounds[1,0], sVals[0] <= bounds[1,1])):
            with hcl.if_(hcl.and_(sVals[2] >= bounds[2,0], sVals[0] <= bounds[2,1])):
                return 1
    return 0


# check if an index is valid
def isIndexValid(iVals, Qopt):
    with hcl.if_(hcl.and_(iVals[0] >= 0, iVals[0] < Qopt.shape[0])):
        with hcl.if_(hcl.and_(iVals[1] >= 0, iVals[1] < Qopt.shape[1])):
            with hcl.if_(hcl.and_(iVals[2] >= 0, iVals[2] < Qopt.shape[2])):
                return 1
    return 0



######################################### VALUE ITERATION ##########################################


def value_iteration_3D():
    def solve_Qopt(Qopt, actions, trans, gamma, epsilon, iVals, sVals, bounds, goals, ptsEachDim, count, maxIters):
            reSweep = hcl.scalar(1, "reSweep")
            oldQ    = hcl.scalar(0, "oldV")
            newQ    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < maxIters[0])):
                reSweep[0] = 0

                # Perform value iteration by sweeping in direction 1
                with hcl.Stage("Sweep_1"):
                    # For all states
                    with hcl.for_(0, Qopt.shape[0], name="i") as i:
                        with hcl.for_(0, Qopt.shape[1], name="j") as j:
                            with hcl.for_(0, Qopt.shape[2], name="k") as k:
                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i,j,k,a]
                                    updateQopt(i, j, k, a, iVals, sVals, Qopt, actions, trans, gamma, bounds, goals, ptsEachDim)
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
                                    updateQopt(i2, j2, k2, a, iVals, sVals, Qopt, actions, trans, gamma, bounds, goals, ptsEachDim)
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
                                    updateQopt(i2, j, k, a, iVals, sVals, Qopt, actions, trans, gamma, bounds, goals, ptsEachDim)
                                    newQ[0] = Qopt[i2,j,k,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1

                # Perform value iteration by sweeping in direction 4
                with hcl.Stage("Sweep_4"):
                    # For all states
                    with hcl.for_(0, Qopt.shape[0], name="i") as i:
                        with hcl.for_(1, Qopt.shape[1] + 1, name="j") as j:
                            with hcl.for_(0, Qopt.shape[2], name="k") as k:
                                j2 = Qopt.shape[0] - j

                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i,j2,k,a]
                                    updateQopt(i, j2, k, a, iVals, sVals, Qopt, actions, trans, gamma, bounds, goals, ptsEachDim)
                                    newQ[0] = Qopt[i,j2,k,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1

                # Perform value iteration by sweeping in direction 5
                with hcl.Stage("Sweep_5"):
                    # For all states
                    with hcl.for_(0, Qopt.shape[0], name="i") as i:
                        with hcl.for_(0, Qopt.shape[1], name="j") as j:
                            with hcl.for_(1, Qopt.shape[2] + 1, name="k") as k:
                                k2 = Qopt.shape[0] - k

                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i,j,k2,a]
                                    updateQopt(i, j, k2, a, iVals, sVals, Qopt, actions, trans, gamma, bounds, goals, ptsEachDim)
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
                                j2 = Qopt.shape[0] - j

                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i2,j2,k,a]
                                    updateQopt(i2, j2, k, a, iVals, sVals, Qopt, actions, trans, gamma, bounds, goals, ptsEachDim)
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
                                k2 = Qopt.shape[0] - k

                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i2,j,k2,a]
                                    updateQopt(i2, j, k2, a, iVals, sVals, Qopt, actions, trans, gamma, bounds, goals, ptsEachDim)
                                    newQ[0] = Qopt[i2,j,k2,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1

                # Perform value iteration by sweeping in direction 8
                with hcl.Stage("Sweep_8"):
                    # For all states
                    with hcl.for_(0, Qopt.shape[0], name="i") as i:
                        with hcl.for_(1, Qopt.shape[1] + 1, name="j") as j:
                            with hcl.for_(1, Qopt.shape[2] + 1, name="k") as k:
                                j2 = Qopt.shape[0] - j
                                k2 = Qopt.shape[0] - k

                                # For all actions
                                with hcl.for_(0, Qopt.shape[3], name="a") as a:
                                    oldQ[0] = Qopt[i,j2,k2,a]
                                    updateQopt(i, j2, k2, a, iVals, sVals, Qopt, actions, trans, gamma, bounds, goals, ptsEachDim)
                                    newQ[0] = Qopt[i,j2,k2,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1


    ###################################### SETUP PLACEHOLDERS ######################################
    
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    # NOTE: trans is a tensor with size = maximum number of transitions
    # NOTE: intermeds must have size  [# possible actions]
    # NOTE: transition must have size [# possible outcomes, #state dimensions + 1]
    Qopt       = hcl.placeholder(tuple([10, 10, 10, 3]), name="Qopt", dtype=hcl.Float())
    gamma      = hcl.placeholder((0,), "gamma")
    count      = hcl.placeholder((0,), "count")
    epsilon    = hcl.placeholder((0,), "epsilon")
    actions    = hcl.placeholder(tuple(_actions.shape), name="actions", dtype=hcl.Float())
    trans      = hcl.placeholder(tuple(_trans.shape), name="successors", dtype=hcl.Float())
    bounds     = hcl.placeholder(tuple([3, 2]), name="bounds", dtype=hcl.Float())
    goals      = hcl.placeholder(tuple(_goals.shape), name="goals", dtype=hcl.Float())
    ptsEachDim = hcl.placeholder(tuple([3]), name="ptsEachDim", dtype=hcl.Float())
    sVals      = hcl.placeholder(tuple([3]), name="sVals", dtype=hcl.Float())
    iVals      = hcl.placeholder(tuple([3]), name="iVals", dtype=hcl.Float())
    maxIters   = hcl.placeholder((0,), "maxIters")

    # Create a static schedule -- graph
    s = hcl.create_schedule([Qopt, actions, trans, gamma, epsilon, iVals, sVals, bounds, goals, ptsEachDim, count, maxIters], solve_Qopt)
    

    ########################################## INITIALIZE ##########################################

    # Convert the python array to hcl type array
    Q_opt      = hcl.asarray(np.zeros([10, 10, 10, 3]))
    trans      = hcl.asarray(_trans)
    gamma      = hcl.asarray(_gamma)
    epsilon    = hcl.asarray(_epsilon)
    count      = hcl.asarray(np.zeros(1))
    actions    = hcl.asarray(_actions)
    bounds     = hcl.asarray(_bounds)
    goals      = hcl.asarray(_goals)
    ptsEachDim = hcl.asarray(_ptsEachDim)
    sVals      = hcl.asarray(np.zeros([3]))
    iVals      = hcl.asarray(np.zeros([3])) 
    maxIters   = hcl.asarray(_maxIters)


    ######################################### PARALLELIZE ##########################################

    s_1 = solve_Qopt.Sweep_1
    s_2 = solve_Qopt.Sweep_2
    s_3 = solve_Qopt.Sweep_3
    s_4 = solve_Qopt.Sweep_4
    s_5 = solve_Qopt.Sweep_5
    s_6 = solve_Qopt.Sweep_6
    s_7 = solve_Qopt.Sweep_7
    s_8 = solve_Qopt.Sweep_8

    s[s_1].parallel(s_1.i)
    s[s_2].parallel(s_2.i)
    s[s_3].parallel(s_3.i)
    s[s_4].parallel(s_4.i)
    s[s_5].parallel(s_5.i)
    s[s_6].parallel(s_6.i)
    s[s_7].parallel(s_7.i)
    s[s_8].parallel(s_8.i)

    # Use this graph and build an executable
    f = hcl.build(s, target="llvm")


    ########################################### EXECUTE ############################################

    t_s = time.time()
    f(Q_opt, actions, trans, gamma, epsilon, iVals, sVals, bounds, goals, ptsEachDim, count, maxIters)
    t_e = time.time()

    Q = Q_opt.asnumpy()
    c = count.asnumpy()

    print(Q)
    print()
    print("Finished in ", int(c[0]), " iterations")
    print("Took        ", t_e-t_s, " seconds")


# Test function
value_iteration_3D()
