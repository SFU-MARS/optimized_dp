import heterocl as hcl
#from Grid.GridProcessing import Grid
import numpy as np
import time
#from user_definer import *



###################################### USER-DEFINED FUNCTIONS ######################################

# Return the reward for taking action from state
def reward(i, j, k, action):
    # s' = (ia, ja, ka)
    ia = i + action[0]
    ja = j + action[1]
    ka = k + action[2]
    rwd = hcl.scalar(0, "rwd")
    with hcl.if_(hcl.and_(ia == 8, ja == 8, ka == 8)):
        rwd[0] = 100
    with hcl.else_():
        rwd[0] = 1

    return rwd[0]



######################################### HELPER FUNCTIONS #########################################

# Update the value function at position (i,j,k)
# NOTE: Determine if HeteroCL is pass by ref or val
def updateQopt(i, j, k, a, Qopt, actions, gamma):
    # s' = (ia, ja, ka)
    action = actions[a]
    ia = i + action[0]
    ja = j + action[1]
    ka = k + action[2]

    # check if action was legal
    with hcl.if_(hcl.and_(ia < Qopt.shape[0], ja < Qopt.shape[1], ka < Qopt.shape[2])):

        # initialize Qopt[i,j,k,a] with the immediate reward
        r = hcl.scalar(0, "r")
        r[0] = reward(i, j, k, action)
        Qopt[i,j,k,a] = r[0]

        # maximize over Q(s', a')
        # i.e. Compare the Q values at the successor state and choose the largest to use as estimate
        with hcl.for_(0, actions.shape[0], name="a_") as a_:
            with hcl.if_(r + (gamma[0] * Qopt[ia,ja,ka,a_]) > Qopt[i,j,k,a]):
                Qopt[i,j,k,a] = r + (gamma[0] * Qopt[ia,ja,ka,a_])

        return Qopt[i,j,k,a]

# Returns 0 if convergence has been reached
def evaluateConvergence(newQ, oldQ, epsilon, reSweep):
    delta = hcl.scalar(0, "delta")
    # Calculate the difference, if it's negative, make it positive
    delta[0] = newQ[0] - oldQ[0]
    with hcl.if_(delta[0] < 0):
        delta[0] = delta[0] * -1
    with hcl.if_(delta[0] > epsilon[0]):
        reSweep[0] = 1



######################################### VALUE ITERATION ##########################################

# Minh: All functions defined within solve_Vopt needs to be re-written as heteroCL (hcl.while, etc)
# Minh: Also arrays values used and passed into solve_Vopt function needs to be placeholder type
def value_iteration_3D():
    def solve_Qopt(Qopt, actions, gamma, epsilon, count):
            reSweep = hcl.scalar(1, "reSweep")
            oldQ    = hcl.scalar(0, "oldV")
            newQ    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < 500)):
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
                                    updateQopt(i, j, k, a, Qopt, actions, gamma)
                                    newQ[0] = Qopt[i,j,k,a]
                                    evaluateConvergence(newQ, oldQ, epsilon, reSweep)
                    count[0] += 1



    ###################################### SETUP PLACEHOLDERS ######################################
    
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    # NOTE: Input to create_schedule should be hcl.placeholder type
    # These placeholder let the compiler knows size of input beforehand ---> faster execution

    # NOTE: trans is a tensor with size = maximum number of transitions
    # NOTE: intermeds must have size  [# possible actions]
    # NOTE: transition must have size [# possible outcomes, #state dimensions + 1]
    Qopt      = hcl.placeholder(tuple([10, 10, 10, 3]), name="Qopt", dtype=hcl.Float())
    gamma     = hcl.placeholder((1,), "gamma")
    count     = hcl.placeholder((0,), "count")
    epsilon   = hcl.placeholder((0,), "epsilon")
    actions   = hcl.placeholder(tuple([3, 3]), name="actions", dtype=hcl.Float())

    # Create a static schedule -- graph
    s = hcl.create_schedule([Qopt, actions, gamma, epsilon, count], solve_Qopt)
    


    ########################################## INITIALIZE ##########################################

    # Convert the python array to hcl type array
    Q_opt = hcl.asarray(np.zeros([10, 10, 10, 3]))

    # set the actions
    actions    = np.ones([3, 3])
    actions[0] = ([1,0,0])
    actions[1] = ([0,1,0])
    actions[2] = ([0,0,1])

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



    ######################################### PARALLELIZE ##########################################

    # Use this graph and build an executable
    f = hcl.build(s, target="llvm")



    ########################################### EXECUTE ############################################

    # Now use the executable
    t_s = time.time()
    f(Q_opt, actions, gamma, epsilon, count)
    t_e = time.time()

    Q = Q_opt.asnumpy()
    c = count.asnumpy()

    print(Q)
    print()
    print("Finished in ", int(c[0]), " iterations")
    print("Took        ", t_e-t_s, " seconds")


# Test function
value_iteration_3D()
