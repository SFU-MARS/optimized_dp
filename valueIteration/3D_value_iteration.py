import heterocl as hcl
#from Grid.GridProcessing import Grid
import numpy as np
import time
#from user_definer import *



###################################### USER-DEFINED FUNCTIONS ######################################

# Given state and action, return successor states and their probabilities
# TODO: Find way to dynamically allocate space
def transition(i, j, k, ai, aj, ak, trans):
    ia = i + ai
    ja = j + aj
    ka = k + ak
    trans[0, 0] = 0.9
    trans[1, 0] = 0.1
    trans[0, 1] = ia
    trans[0, 2] = ja
    trans[0, 3] = ka
    trans[1, 1] = i
    trans[1, 2] = j
    trans[1, 3] = k

# Return the reward for taking action from state
def reward(i, j, k, action):
    rwd = hcl.scalar(0, "rwd")
    with hcl.if_(hcl.and_(i == 8, j == 8, k == 8)):
        rwd[0] = 100
    with hcl.else_():
        rwd[0] = 1

    return rwd[0]


######################################### HELPER FUNCTIONS #########################################

# Update the value function at position (i,j,k)
# NOTE: Determine if HeteroCL is pass by ref or val
# TODO: Make use of probability weights
def updateVopt(i, j, k, actions, Vopt, intermeds, trans, gamma):
    with hcl.for_(0, actions.shape[0], name="a") as a:
        transition(i,j,k, actions[a, 0], actions[a, 1], actions[a, 2], trans)
        intermeds[a] = reward(i, j, k, actions[a])

        # Calculate resulting values
        with hcl.for_(0, trans.shape[0], name="si") as si:
            ia = trans[si,1]
            ja = trans[si,2]
            ka = trans[si,3]

            # check boundaries
            with hcl.if_(hcl.and_(ia < Vopt.shape[0], ja < Vopt.shape[1], ka < Vopt.shape[2])):
                intermeds[a] += (gamma[0] * (trans[si, 0] * Vopt[trans[si,1], trans[si,2], trans[si,3]]))

    # Maximize over the intermediates
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


######################################### VALUE ITERATION ##########################################

# Minh: All functions defined within solve_Vopt needs to be re-written as heteroCL (hcl.while, etc)
# Minh: Also arrays values used and passed into solve_Vopt function needs to be placeholder type
def value_iteration_3D():
    # Perform value iteration by sweeping in direction 1
    def solve_Vopt(Vopt, actions, intermeds, trans, gamma, epsilon, count):
        with hcl.Stage("Sweep_1"):
            reSweep = hcl.scalar(1, "reSweep")
            oldV    = hcl.scalar(0, "oldV")
            newV    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < 500000)):
                reSweep[0] = 0
                with hcl.for_(0, Vopt.shape[0], name="i") as i:
                    with hcl.for_(0, Vopt.shape[1], name="j") as j:
                        with hcl.for_(0, Vopt.shape[2], name="k") as k:
                            oldV[0] = Vopt[i,j,k]
                            updateVopt(i, j, k, actions, Vopt, intermeds, trans, gamma)
                            newV[0] = Vopt[i,j,k]
                            evaluateConvergence(newV, oldV, epsilon, reSweep)
                            count[0] += 1

    ############################################ SETUP #############################################
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    # NOTE: Input to create_schedule should be hcl.placeholder type
    # These placeholder let the compiler knows size of input beforehand ---> faster execution
    Vopt    = hcl.placeholder(tuple([10, 10, 10]), name="Vopt", dtype=hcl.Float())
    gamma   = hcl.placeholder((1,), "gamma")
    count   = hcl.placeholder((0,), "count")
    epsilon = hcl.placeholder((0,), "epsilon")

    # TODO: Implement the transitions as a tensor here with size = maximum number of transitions
    trans = hcl.placeholder(tuple([2,4]), name="successors", dtype=hcl.Float())

    # Example of actions placeholder, rwd placeholder and transition placeholder
    actions   = hcl.placeholder(tuple([3, 3]), name="actions", dtype=hcl.Float())
    intermeds = hcl.placeholder(tuple([5]), name="intermeds", dtype=hcl.Float())
    rwd       = hcl.placeholder(tuple([5]), name="rwd", dtype=hcl.Float())

    # Create a static schedule -- graph
    s = hcl.create_schedule([Vopt, actions, intermeds, trans, gamma, epsilon, count], solve_Vopt)
    
    # Use this graph and build an executable
    f = hcl.build(s)

    # Now use this executable for our Python program

    # Convert the python array to hcl type array
    V_opt     = hcl.asarray(np.zeros([10, 10, 10]))
    intermeds = hcl.asarray(np.ones([3]))
    trans     = hcl.asarray(np.zeros([2,4]))
    rwd       = np.zeros([5])

    # set the actions
    actions = np.ones([5, 3])
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

    # Now use the executable
    f(V_opt, actions, intermeds, trans, gamma, epsilon, count)

    V = V_opt.asnumpy()
    c = count.asnumpy()

    print(V)
    print()
    print("# Irerations: ", int(c[0]/1000))

# Test function
value_iteration_3D()