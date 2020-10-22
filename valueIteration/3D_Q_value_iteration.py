import heterocl as hcl
#from Grid.GridProcessing import Grid
import numpy as np
import time
#from user_definer import *



###################################### USER-DEFINED FUNCTIONS ######################################

# Given state and action, return successor states and their probabilities
def transition(i, j, k, action, trans):
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
def reward(i, j, k, Qopt, action, trans):
    rwd = hcl.scalar(0, "rwd")
    # for each possible successor
    with hcl.for_(0, trans.shape[0], name="si") as si:
        p  = trans[si,0]
        ia = trans[si,1]
        ja = trans[si,2]
        ka = trans[si,3]
        # check if action was legal
        with hcl.if_(hcl.and_(ia < Qopt.shape[0], ja < Qopt.shape[1], ka < Qopt.shape[2])):
            # add reward for reaching goal
            with hcl.if_(hcl.and_(ia == 8, ja == 8, ka == 8)):
                rwd[0] += (p * 100)
            # add default reward
            with hcl.else_():
                rwd[0] += 1

    return rwd[0]



######################################### HELPER FUNCTIONS #########################################

# Update the value function at position (i,j,k)
def updateQopt(i, j, k, a, Qopt, actions, trans, gamma):
    # load successors
    transition(i, j, k, actions[a], trans)

    # initialize Qopt[i,j,k,a] with the immediate reward
    r = hcl.scalar(0, "r")
    r[0] = reward(i, j, k, Qopt, actions[a], trans)
    Qopt[i,j,k,a] = r[0]

    # maximize over successor Q-values
    with hcl.for_(0, trans.shape[0], name="si") as si:
        p    = trans[si,0]
        ia   = trans[si,1]
        ja   = trans[si,2]
        ka   = trans[si,3]

        # check if successor is within the grid
        with hcl.if_(hcl.and_(ia < Qopt.shape[0], ja < Qopt.shape[1], ka < Qopt.shape[2])):
            with hcl.for_(0, actions.shape[0], name="a_") as a_:
                with hcl.if_(r + (gamma[0] * (p * Qopt[ia,ja,ka,a_])) > Qopt[i,j,k,a]):
                    Qopt[i,j,k,a] = r + (gamma[0] * (p * Qopt[ia,ja,ka,a_]))
            r = Qopt[i,j,k,a] 

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

def value_iteration_3D():
    def solve_Qopt(Qopt, actions, trans, gamma, epsilon, count):
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
                                    updateQopt(i, j, k, a, Qopt, actions, trans, gamma)
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
                                    updateQopt(i2, j2, k2, a, Qopt, actions, trans, gamma)
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
                                    updateQopt(i2, j, k, a, Qopt, actions, trans, gamma)
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
                                    updateQopt(i, j2, k, a, Qopt, actions, trans, gamma)
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
                                    updateQopt(i, j, k2, a, Qopt, actions, trans, gamma)
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
                                    updateQopt(i2, j2, k, a, Qopt, actions, trans, gamma)
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
                                    updateQopt(i2, j, k2, a, Qopt, actions, trans, gamma)
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
                                    updateQopt(i, j2, k2, a, Qopt, actions, trans, gamma)
                                    newQ[0] = Qopt[i,j2,k2,a]
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
    trans     = hcl.placeholder(tuple([2,4]), name="successors", dtype=hcl.Float())

    # Create a static schedule -- graph
    s = hcl.create_schedule([Qopt, actions, trans, gamma, epsilon, count], solve_Qopt)
    


    ########################################## INITIALIZE ##########################################

    # Convert the python array to hcl type array
    Q_opt = hcl.asarray(np.zeros([10, 10, 10, 3]))
    trans = hcl.asarray(np.zeros([2, 4]))

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

    # Now use the executable
    t_s = time.time()
    f(Q_opt, actions, trans, gamma, epsilon, count)
    t_e = time.time()

    Q = Q_opt.asnumpy()
    c = count.asnumpy()

    print(Q)
    print()
    print("Finished in ", int(c[0]), " iterations")
    print("Took        ", t_e-t_s, " seconds")


# Test function
value_iteration_3D()
