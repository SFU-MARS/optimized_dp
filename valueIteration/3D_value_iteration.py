import heterocl as hcl
#from Grid.GridProcessing import Grid
import numpy as np
import time
#from user_definer import *



###################################### USER-DEFINED FUNCTIONS ######################################

# Given state and action, return successor states and their probabilities
# 
def transition(i, j, k, action):
    scr = hcl.asarray(np.zeros([2, 4]))
    ia  = i + action[0]
    ja  = j + action[1]
    ka  = k + action[2]
    scr[0, 0] = 0.6
    scr[1, 0] = 0.4
    scr[0, 1] = i + ia
    scr[0, 2] = i + ja
    scr[0, 3] = i + ka
    scr[1, 1] = i
    scr[1, 2] = j
    scr[1, 3] = k
    return scr

# Return the reward for taking action from state
def reward(i, j, k, action):
    rwd = hcl.scalar(0, "rwd")
    with hcl.if_(hcl.and_(i == 8, j == 8, k == 8)):
        rwd[0] = 8
    with hcl.else_():
        rwd[0] = 1

    return rwd[0]


######################################### HELPER FUNCTIONS #########################################

# Update the value function at position (i,j,k)
# NOTE: Determine if HeteroCL is pass by ref or val
# TODO: Make use of probability weights
def updateVopt(i, j, k, actions, Vopt, intermeds):
    with hcl.for_(0, actions.shape[0], name="a") as a:
        ia = i + actions[a, 0]
        ja = j + actions[a, 1]
        ka = k + actions[a, 2]
        # Calculate resulting value
        with hcl.if_(hcl.and_(ia < Vopt.shape[0], ja < Vopt.shape[1], ka < Vopt.shape[2])):
            intermeds[a] = Vopt[ia,ja,ka] + reward(i, j, k, a)
    # Maximize over the intermediates
    with hcl.for_(0, intermeds.shape[0], name="r") as r:
        with hcl.if_(Vopt[i,j,k] < intermeds[r]):
            Vopt[i,j,k] = intermeds[r]
    return Vopt[i,j,k]

# Returns 0 if convergence has been reached
def evaluateConvergence(newV, oldV, epsilon):
    with hcl.if_(hcl.or_((newV - oldV) > epsilon[0], (oldV - newV) > epsilon[0])):
        return 1
    with hcl.else_():
        return 0


######################################### VALUE ITERATION ##########################################

# Minh: All functions defined within solve_Vopt needs to be re-written as heteroCL (hcl.while, etc)
# Minh: Also arrays values used and passed into solve_Vopt function needs to be placeholder type
def value_iteration_3D():
    # Perform value iteration by sweeping in direction 1
    def solve_Vopt(Vopt, actions, intermeds, trans, gamma, epsilon):
        with hcl.Stage("Sweep_1"):
            reSweep = hcl.scalar(1, "reSweep")
            with hcl.while_(reSweep[0] == 1):
                reSweep[0] = 0
                with hcl.for_(0, Vopt.shape[0], name="i") as i:
                    with hcl.for_(0, Vopt.shape[1], name="j") as j:
                        with hcl.for_(0, Vopt.shape[2], name="k") as k:
                            oldV    = Vopt[i,j,k]
                            newV    = updateVopt(i, j, k, actions, Vopt, intermeds)
                            reSweep = evaluateConvergence(newV, oldV, epsilon)

    # Setup
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    # NOTE: Input to create_schedule should be hcl.placeholder type
    # These placeholder let the compiler knows size of input beforehand ---> faster execution
    Vinit   = hcl.placeholder(tuple([10, 10, 10]), name="Vinit", dtype=hcl.Float())
    Vopt    = hcl.placeholder(tuple([10, 10, 10]), name="Vopt", dtype=hcl.Float())
    gamma   = hcl.placeholder((1,), "gamma")
    epsilon = hcl.placeholder((0,), "epsilon")
    reSweep = hcl.placeholder((1,), "reSweep")

    # Example of actions placeholder, rwd placeholder and transition placeholder
    actions   = hcl.placeholder(tuple([5, 3]), name="actions", dtype=hcl.Float())
    intermeds = hcl.placeholder(tuple([5]), name="intermeds", dtype=hcl.Float())
    rwd       = hcl.placeholder(tuple([5]), name="rwd", dtype=hcl.Float())
    trans     = hcl.placeholder(tuple([5]), name="trans", dtype=hcl.Float())

    # Create a static schedule -- graph
    s = hcl.create_schedule([Vopt, actions, intermeds, trans, gamma, epsilon], solve_Vopt)
    
    # Use this graph and build an executable
    f = hcl.build(s)

    # Now use this executable for our Python program

    # Convert the python array to hcl type array
    V_init    = hcl.asarray(np.zeros([10, 10, 10]))
    V_opt     = hcl.asarray(np.zeros([10, 10, 10]))
    actions   = hcl.asarray(np.ones([5, 3]))
    intermeds = hcl.asarray(np.ones([5]))
    trans     = hcl.asarray(np.zeros([5]))
    rwd       = np.zeros([5])

    # set the trans functions
    rwd[3]    = 2
    rwd[1]    = 3
    rwd       = hcl.asarray(trans)

    gamma     = hcl.asarray(np.zeros(1))
    epsilon   = hcl.asarray(np.zeros(1))

    # Now use the executable
    f(V_opt, actions, intermeds, trans, gamma, epsilon)

    print(V_opt)

# Test function
value_iteration_3D()