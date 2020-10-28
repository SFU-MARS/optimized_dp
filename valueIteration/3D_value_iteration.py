import heterocl as hcl
#from Grid.GridProcessing import Grid
import numpy as np
import time
#from user_definer import *



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
def reward(sVals, action):
    rwd = hcl.scalar(0, "rwd")
    with hcl.if_(hcl.and_(sVals[0] == 8, sVals[1] == 8, sVals[2] == 8)): rwd[0] = 100
    with hcl.else_():                                                    rwd[0] = 1
    return rwd[0]



######################################### HELPER FUNCTIONS #########################################


# Update the value function at position (i,j,k)
def updateVopt(i, j, k, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, ptsEachDim):
    with hcl.for_(0, actions.shape[0], name="a") as a:
        # set iVals equal to (i,j,k) and sVals equal to the corresponding state values at (i,j,k)
        updateStateVals(i, j, k, iVals, sVals, bounds, ptsEachDim)
        # call the transition function to obtain the outcome(s) of action a from state (si,sj,sk)
        transition(sVals, actions[a], trans)
        # initialize the value of the action using the immediate reward of taking that action
        intermeds[a] = reward(sVals, actions[a])
        # add the value of each possible successor state
        with hcl.for_(0, trans.shape[0], name="si") as si:
            p        = trans[si,0]
            sVals[0] = trans[si,1]
            sVals[1] = trans[si,2]
            sVals[2] = trans[si,3]
            # convert the state values of the success state (si,sj,sk) into indeces (ia, ij, ik)
            stateToIndex(sVals, iVals, bounds, ptsEachDim)
            # if (ia, ij, ik) is within the state space, add its discounted value to action a
            with hcl.if_(hcl.and_(iVals[0] < Vopt.shape[0], iVals[1] < Vopt.shape[1], iVals[2] < Vopt.shape[2])):
                with hcl.if_(hcl.and_(iVals[0] >= 0, iVals[1] >= 0, iVals[2] >= 0)):
                    intermeds[a] += (gamma[0] * (p * Vopt[iVals[0], iVals[1], iVals[2]]))
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


# convert state values into indeces using nearest neighbour
def stateToIndex(sVals, iVals, bounds, ptsEachDim):
    iVals[0] = (sVals[0] - bounds[0,0]) / (bounds[0,1] - bounds[0,0]) * ptsEachDim[0]
    iVals[1] = (sVals[1] - bounds[1,0]) / (bounds[1,1] - bounds[1,0]) * ptsEachDim[1]
    iVals[2] = (sVals[2] - bounds[2,0]) / (bounds[2,1] - bounds[2,0]) * ptsEachDim[2]
    # NOTE: add 0.5 to simulate rounding
    iVals[0] = hcl.cast(hcl.Int(), iVals[0] + 0.5)
    iVals[1] = hcl.cast(hcl.Int(), iVals[1] + 0.5)
    iVals[2] = hcl.cast(hcl.Int(), iVals[2] + 0.5)


# convert indices into state values
def indexToState(iVals, sVals, bounds, ptsEachDim): 
    sVals[0] = (iVals[0] / ptsEachDim[0]) * (bounds[0,1] - bounds[0,0]) + bounds[0,0]
    sVals[1] = (iVals[1] / ptsEachDim[1]) * (bounds[1,1] - bounds[1,0]) + bounds[1,0]
    sVals[2] = (iVals[2] / ptsEachDim[2]) * (bounds[2,1] - bounds[2,0]) + bounds[2,0]


def updateStateVals(i, j, k, iVals, sVals, bounds, ptsEachDim):
    iVals[0] = i
    iVals[1] = j
    iVals[2] = k
    indexToState(iVals, sVals, bounds, ptsEachDim)



######################################### VALUE ITERATION ##########################################


# Minh: All functions defined within solve_Vopt needs to be re-written as heteroCL (hcl.while, etc)
# Minh: Also arrays values used and passed into solve_Vopt function needs to be placeholder type
def value_iteration_3D():
    def solve_Vopt(Vopt, actions, intermeds, trans, gamma, epsilon, iVals, sVals, bounds, ptsEachDim, count):
            reSweep = hcl.scalar(1, "reSweep")
            oldV    = hcl.scalar(0, "oldV")
            newV    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < 500)):
                reSweep[0] = 0

                # Perform value iteration by sweeping in direction 1
                with hcl.Stage("Sweep_1"):
                    with hcl.for_(0, Vopt.shape[0], name="i") as i:
                        with hcl.for_(0, Vopt.shape[1], name="j") as j:
                            with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                oldV[0] = Vopt[i,j,k]
                                updateVopt(i, j, k, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, ptsEachDim)
                                newV[0] = Vopt[i,j,k]
                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1

                # Perform value iteration by sweeping in direction 2
                with hcl.Stage("Sweep_2"):
                    with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                        with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                            with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                i2 = Vopt.shape[0] - i
                                j2 = Vopt.shape[1] - j
                                k2 = Vopt.shape[2] - k
                                oldV[0] = Vopt[i2,j2,k2]
                                updateVopt(i2, j2, k2, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, ptsEachDim)
                                newV[0] = Vopt[i2,j2,k2]
                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1

                # Perform value iteration by sweeping in direction 3
                with hcl.Stage("Sweep_3"):
                    with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                        with hcl.for_(0, Vopt.shape[1], name="j") as j:
                            with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                i2 = Vopt.shape[0] - i
                                oldV[0] = Vopt[i2,j,k]
                                updateVopt(i2, j, k, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, ptsEachDim)
                                newV[0] = Vopt[i2,j,k]
                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1

                # Perform value iteration by sweeping in direction 4
                with hcl.Stage("Sweep_4"):
                    with hcl.for_(0, Vopt.shape[0], name="i") as i:
                        with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                            with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                j2 = Vopt.shape[1] - j
                                oldV[0] = Vopt[i,j2,k]
                                updateVopt(i, j2, k, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, ptsEachDim)
                                newV[0] = Vopt[i,j2,k]
                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1

                # Perform value iteration by sweeping in direction 5
                with hcl.Stage("Sweep_5"):
                    with hcl.for_(0, Vopt.shape[0], name="i") as i:
                        with hcl.for_(0, Vopt.shape[1], name="j") as j:
                            with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                k2 = Vopt.shape[2] - k
                                oldV[0] = Vopt[i,j,k2]
                                updateVopt(i, j, k2, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, ptsEachDim)
                                newV[0] = Vopt[i,j,k2]
                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1

                # Perform value iteration by sweeping in direction 6
                with hcl.Stage("Sweep_6"):
                    with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                        with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                            with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                i2 = Vopt.shape[0] - i
                                j2 = Vopt.shape[1] - j
                                oldV[0] = Vopt[i2,j2,k]
                                updateVopt(i2, j2, k, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, ptsEachDim)
                                newV[0] = Vopt[i2,j2,k]
                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1

                # Perform value iteration by sweeping in direction 7
                with hcl.Stage("Sweep_7"):
                    with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                        with hcl.for_(0, Vopt.shape[1], name="j") as j:
                            with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                i2 = Vopt.shape[0] - i
                                k2 = Vopt.shape[2] - k
                                oldV[0] = Vopt[i2,j,k2]
                                updateVopt(i2, j, k2, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, ptsEachDim)
                                newV[0] = Vopt[i2,j,k2]
                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1

                # Perform value iteration by sweeping in direction 8
                with hcl.Stage("Sweep_8"):
                    with hcl.for_(0, Vopt.shape[0], name="i") as i:
                        with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                            with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                                j2 = Vopt.shape[1] - j
                                k2 = Vopt.shape[2] - k
                                oldV[0] = Vopt[i,j2,k2]
                                updateVopt(i, j2, k2, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, ptsEachDim)
                                newV[0] = Vopt[i,j2,k2]
                                evaluateConvergence(newV, oldV, epsilon, reSweep)
                    count[0] += 1



    ###################################### SETUP PLACEHOLDERS ######################################
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    # NOTE: Input to create_schedule should be hcl.placeholder type
    # These placeholder let the compiler knows size of input beforehand ---> faster execution

    # NOTE: trans is a tensor with size = maximum number of transitions
    # NOTE: intermeds must have size  [# possible actions]
    # NOTE: transition must have size [# possible outcomes, #state dimensions + 1]
    Vopt      = hcl.placeholder(tuple([30, 30, 30]), name="Vopt", dtype=hcl.Float())
    gamma     = hcl.placeholder((1,), "gamma")
    count     = hcl.placeholder((0,), "count")
    epsilon   = hcl.placeholder((0,), "epsilon")
    actions   = hcl.placeholder(tuple([3, 3]), name="actions", dtype=hcl.Float())
    intermeds = hcl.placeholder(tuple([3]), name="intermeds", dtype=hcl.Float())
    trans     = hcl.placeholder(tuple([2,4]), name="successors", dtype=hcl.Float())
    rwd       = hcl.placeholder(tuple([5]), name="rwd", dtype=hcl.Float())

    # Required for using true state values
    bounds     = hcl.placeholder(tuple([3, 2]), name="bounds", dtype=hcl.Float())
    ptsEachDim = hcl.placeholder(tuple([3]), name="ptsEachDim", dtype=hcl.Float())
    sVals      = hcl.placeholder(tuple([3]), name="sVals", dtype=hcl.Float())
    iVals      = hcl.placeholder(tuple([3]), name="iVals", dtype=hcl.Float())


    # Create a static schedule -- graph
    s = hcl.create_schedule([Vopt, actions, intermeds, trans, gamma, epsilon, iVals, sVals, bounds, ptsEachDim, count], solve_Vopt)
    


    ########################################## INITIALIZE ##########################################

    # Convert the python array to hcl type array
    V_opt     = hcl.asarray(np.zeros([30, 30, 30]))
    intermeds = hcl.asarray(np.ones([3]))
    trans     = hcl.asarray(np.zeros([2, 4]))
    rwd       = np.zeros([5])

    # set the actions
    actions = np.ones([3, 3])
    actions[0] = ([1,0,0])
    actions[1] = ([0,1,0])
    actions[2] = ([0,0,1])

    # set the rwd functions
    rwd[3]    = 2
    rwd[1]    = 3
    rwd       = hcl.asarray(rwd)

    # set the gamma value
    gamma     = np.zeros(1)
    gamma[0]  = .9
    
    # set the epsilon value
    epsilon    = np.zeros(1)
    epsilon[0] = .0000005

    gamma     = hcl.asarray(gamma)
    epsilon   = hcl.asarray(epsilon)
    count     = hcl.asarray(np.zeros(1))
    actions   = hcl.asarray(actions)

    # Required for using true state values
    bounds     = hcl.asarray( np.array([[-5.0, 5.0],[-3.0, 3.0],[10, 15]]) )
    ptsEachDim = hcl.asarray( np.array([40, 40, 50]) )
    sVals      = hcl.asarray( np.zeros([3]) )
    iVals      = hcl.asarray( np.zeros([3]) )


    ######################################### PARALLELIZE ##########################################

    s_1 = solve_Vopt.Sweep_1
    s_2 = solve_Vopt.Sweep_2
    s_3 = solve_Vopt.Sweep_3
    s_4 = solve_Vopt.Sweep_4
    s_5 = solve_Vopt.Sweep_5
    s_6 = solve_Vopt.Sweep_6
    s_7 = solve_Vopt.Sweep_7
    s_8 = solve_Vopt.Sweep_8

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

    print(hcl.lower(s))


    ########################################### EXECUTE ############################################

    # Now use the executable
    t_s = time.time()
    f(V_opt, actions, intermeds, trans, gamma, epsilon, iVals, sVals, bounds, ptsEachDim, count)
    t_e = time.time()

    V = V_opt.asnumpy()
    c = count.asnumpy()

    print(V)
    print()
    print("Finished in ", int(c[0]), " iterations")
    print("Took        ", t_e-t_s, " seconds")


# Test function
value_iteration_3D()
