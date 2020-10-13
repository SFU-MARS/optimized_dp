import heterocl as hcl
#from Grid.GridProcessing import Grid
import numpy as np
import time
#from user_definer import *



###################################### USER-DEFINED FUNCTIONS ######################################

# Given state and action, return successor states and their probabilities
# TODO: Find way to dynamically allocate space
def transition(i, j , k, action, trans):
    ia = i + action[0]
    ja = j + action[1]
    ka = k + action[2]
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
def updateVopt(i, j, k, actions, Vopt, intermeds, trans, gamma):
    with hcl.for_(0, actions.shape[0], name="a") as a:
        transition(i,j,k, actions[a], trans)
        intermeds[a] = reward(i, j, k, actions[a])

        # Calculate resulting values
        with hcl.for_(0, trans.shape[0], name="si") as si:
            p  = trans[si,0]
            ia = trans[si,1]
            ja = trans[si,2]
            ka = trans[si,3]

            # check boundaries
            with hcl.if_(hcl.and_(ia < Vopt.shape[0], ja < Vopt.shape[1], ka < Vopt.shape[2])):
                with hcl.if_(hcl.and_(ia >= 0, ja >= 0, ka >= 0)):
                    intermeds[a] += (gamma[0] * (p * Vopt[ia, ja, ka]))

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
    def solve_Vopt(Vopt, actions, intermeds, trans, gamma, epsilon, count):

        # Perform value iteration by sweeping in direction 1
        with hcl.Stage("Sweep_1"):
            reSweep = hcl.scalar(1, "reSweep")
            oldV    = hcl.scalar(0, "oldV")
            newV    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < 500)):
                reSweep[0] = 0
                with hcl.for_(0, Vopt.shape[0], name="i") as i:
                    with hcl.for_(0, Vopt.shape[1], name="j") as j:
                        with hcl.for_(0, Vopt.shape[2], name="k") as k:
                            oldV[0] = Vopt[i,j,k]
                            updateVopt(i, j, k, actions, Vopt, intermeds, trans, gamma)
                            newV[0] = Vopt[i,j,k]
                            evaluateConvergence(newV, oldV, epsilon, reSweep)
                count[0] += 1

        # Perform value iteration by sweeping in direction 2
        with hcl.Stage("Sweep_2"):
            reSweep = hcl.scalar(1, "reSweep")
            oldV    = hcl.scalar(0, "oldV")
            newV    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < 500)):
                reSweep[0] = 0
                with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                    with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                        with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                            i2 = Vopt.shape[0] - i
                            j2 = Vopt.shape[1] - j
                            k2 = Vopt.shape[2] - k
                            oldV[0] = Vopt[i2,j2,k2]
                            updateVopt(i2, j2, k2, actions, Vopt, intermeds, trans, gamma)
                            newV[0] = Vopt[i2,j2,k2]
                            evaluateConvergence(newV, oldV, epsilon, reSweep)
                count[0] += 0

        # Perform value iteration by sweeping in direction 3
        with hcl.Stage("Sweep_3"):
            reSweep = hcl.scalar(1, "reSweep")
            oldV    = hcl.scalar(0, "oldV")
            newV    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < 500)):
                reSweep[0] = 0
                with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                    with hcl.for_(0, Vopt.shape[1], name="j") as j:
                        with hcl.for_(0, Vopt.shape[2], name="k") as k:
                            i2 = Vopt.shape[0] - i
                            oldV[0] = Vopt[i2,j,k]
                            updateVopt(i2, j, k, actions, Vopt, intermeds, trans, gamma)
                            newV[0] = Vopt[i2,j,k]
                            evaluateConvergence(newV, oldV, epsilon, reSweep)
                count[0] += 0


        # Perform value iteration by sweeping in direction 4
        with hcl.Stage("Sweep_4"):
            reSweep = hcl.scalar(1, "reSweep")
            oldV    = hcl.scalar(0, "oldV")
            newV    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < 500)):
                reSweep[0] = 0
                with hcl.for_(0, Vopt.shape[0], name="i") as i:
                    with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                        with hcl.for_(0, Vopt.shape[2], name="k") as k:
                            j2 = Vopt.shape[1] - j
                            oldV[0] = Vopt[i,j2,k]
                            updateVopt(i, j2, k, actions, Vopt, intermeds, trans, gamma)
                            newV[0] = Vopt[i,j2,k]
                            evaluateConvergence(newV, oldV, epsilon, reSweep)
                count[0] += 0    

        # Perform value iteration by sweeping in direction 5
        # TODO: Check indeces
        with hcl.Stage("Sweep_5"):
            reSweep = hcl.scalar(1, "reSweep")
            oldV    = hcl.scalar(0, "oldV")
            newV    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < 500)):
                reSweep[0] = 0
                with hcl.for_(0, Vopt.shape[0], name="i") as i:
                    with hcl.for_(0, Vopt.shape[1], name="j") as j:
                        with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                            k2 = Vopt.shape[2] - k
                            oldV[0] = Vopt[i,j,k2]
                            updateVopt(i, j, k2, actions, Vopt, intermeds, trans, gamma)
                            newV[0] = Vopt[i,j,k2]
                            evaluateConvergence(newV, oldV, epsilon, reSweep)
                count[0] += 0   

        # Perform value iteration by sweeping in direction 6
        with hcl.Stage("Sweep_6"):
            reSweep = hcl.scalar(1, "reSweep")
            oldV    = hcl.scalar(0, "oldV")
            newV    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < 500)):
                reSweep[0] = 0
                with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                    with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                        with hcl.for_(0, Vopt.shape[2], name="k") as k:
                            i2 = Vopt.shape[0] - i
                            j2 = Vopt.shape[1] - j
                            oldV[0] = Vopt[i2,j2,k]
                            updateVopt(i2, j2, k, actions, Vopt, intermeds, trans, gamma)
                            newV[0] = Vopt[i2,j2,k]
                            evaluateConvergence(newV, oldV, epsilon, reSweep)
                count[0] += 0
                
        # Perform value iteration by sweeping in direction 7
        with hcl.Stage("Sweep_7"):
            reSweep = hcl.scalar(1, "reSweep")
            oldV    = hcl.scalar(0, "oldV")
            newV    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < 500)):
                reSweep[0] = 0
                with hcl.for_(1, Vopt.shape[0] + 1, name="i") as i:
                    with hcl.for_(0, Vopt.shape[1], name="j") as j:
                        with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                            i2 = Vopt.shape[0] - i
                            k2 = Vopt.shape[2] - k
                            oldV[0] = Vopt[i2,j,k2]
                            updateVopt(i2, j, k2, actions, Vopt, intermeds, trans, gamma)
                            newV[0] = Vopt[i2,j,k2]
                            evaluateConvergence(newV, oldV, epsilon, reSweep)
                count[0] += 0

        # Perform value iteration by sweeping in direction 8
        with hcl.Stage("Sweep_8"):
            reSweep = hcl.scalar(1, "reSweep")
            oldV    = hcl.scalar(0, "oldV")
            newV    = hcl.scalar(0, "newV")
            with hcl.while_(hcl.and_(reSweep[0] == 1, count[0] < 500)):
                reSweep[0] = 0
                with hcl.for_(0, Vopt.shape[0], name="i") as i:
                    with hcl.for_(1, Vopt.shape[1] + 1, name="j") as j:
                        with hcl.for_(1, Vopt.shape[2] + 1, name="k") as k:
                            j2 = Vopt.shape[1] - j
                            k2 = Vopt.shape[2] - k
                            oldV[0] = Vopt[i,j2,k2]
                            updateVopt(i, j2, k2, actions, Vopt, intermeds, trans, gamma)
                            newV[0] = Vopt[i,j2,k2]
                            evaluateConvergence(newV, oldV, epsilon, reSweep)
                count[0] += 0



    ###################################### SETUP PLACEHOLDERS ######################################
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    # NOTE: Input to create_schedule should be hcl.placeholder type
    # These placeholder let the compiler knows size of input beforehand ---> faster execution

    # NOTE: trans is a tensor with size = maximum number of transitions
    # NOTE: intermeds must have size  [# possible actions]
    # NOTE: transition must have size [# possible outcomes, #state dimensions + 1]
    Vopt      = hcl.placeholder(tuple([10, 10, 10]), name="Vopt", dtype=hcl.Float())
    gamma     = hcl.placeholder((1,), "gamma")
    count     = hcl.placeholder((0,), "count")
    epsilon   = hcl.placeholder((0,), "epsilon")
    actions   = hcl.placeholder(tuple([3, 3]), name="actions", dtype=hcl.Float())
    intermeds = hcl.placeholder(tuple([3]), name="intermeds", dtype=hcl.Float())
    trans     = hcl.placeholder(tuple([2,4]), name="successors", dtype=hcl.Float())
    rwd       = hcl.placeholder(tuple([5]), name="rwd", dtype=hcl.Float())

    # Create a static schedule -- graph
    s = hcl.create_schedule([Vopt, actions, intermeds, trans, gamma, epsilon, count], solve_Vopt)
    


    ########################################## INITIALIZE ##########################################

    # Convert the python array to hcl type array
    V_opt     = hcl.asarray(np.zeros([10, 10, 10]))
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
    f(V_opt, actions, intermeds, trans, gamma, epsilon, count)

    V = V_opt.asnumpy()
    c = count.asnumpy()

    print(V)
    print()
    print("# Irerations: ", int(c[0]))

# Test function
value_iteration_3D()
