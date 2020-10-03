import heterocl as hcl
#from Grid.GridProcessing import Grid
import numpy as np
import time
#from user_definer import *

# Perform value iteration to solve for the optimal value function, Vopt
# grid:       a Grid object characterizing the state space
# actions:    the set of actions a that can be taken from any state 
#             TODO: must be in tensor format
# rwd:        reward for being in state s and taking action a
#             TODO: must be in tensor format
# trans:      given state s, action a, return the set of successor states s' and
#             the associated probability p of reaching s'
#             TODO: how to represent the transition function as a tensor?
# discount:   discount factor
# epsilon:    a small number to guage congergence

def value_iteration_3D():
    # Minh: All functions defined within solve_Vopt needs to be re-written as heteroCL (hcl.while, etc)
    # Minh: Also arrays values used and passed into solve_Vopt function needs to be placeholder type
    def solve_Vopt(Vinit, Vopt, actions, rwd, trans, gamma, epsilon, g_vs1, g_vs2, g_vs3):
        # Vopt = hcl.asarray(Vinit)
        # discount = gamma
        # epsilon  = convergence_number

        # Calculate the value of Vopt at point s given action a
        # TODO: Is it possible to make calls to an actions(s,a) or trans(s,a) func?
        # TODO: Rewrite with heteroCL syntax
        # def updateV(s):
        #     return max(rwd(s, a) +
        #         discount*sum(p*Vopt[index_of_state(sNew)] for p, sNew in trans(s,a)) for a in actions)


        # Return the index values associated with the given stateValues
        # TODO: Find better way of obtaining the index from a set of state vals
        # TODO: Rewrite with heteroCL syntax
        # def index_of_state(stateVals, grid):
        #     i = round(stateVals[0] * grid.pts_each_dim[0] / (grid.max[0] - grid.min[0]))
        #     j = round(stateVals[1] * grid.pts_each_dim[1] / (grid.max[1] - grid.min[1]))
        #     k = round(stateVals[2] * grid.pts_each_dim[2] / (grid.max[2] - grid.min[2]))
        #     return (i, j, k)

        with hcl.Stage("Sweep_1"):
            reSweep = hcl.scalar(1, "reSweep")
            with hcl.while_(reSweep == 1):
                reSweep[0] = 0
                with hcl.for_(0, 10, name="i") as i:
                    with hcl.for_(0, 10, name="j") as j:
                        with hcl.for_(0, 10, name="k") as k:
                            s = hcl.placeholder((1,3), "state")
                            oldV = hcl.scalar(0, "oldV")
                            newV = hcl.scalar(0, "newV")
                            # Calculate the new optimal value
                            #s = (g.vs[0][i,0,0], g.vs[1][0,j,0], g.vs[2][0,0,k])
                            s = (g_vs1[i], g_vs2[j], g_vs3[k])
                            oldV[0] = Vinit[i, j, k]
                            #newV = updateV(s)
                            # Sample update
                            newV[0] = oldV[0] + 1
                            Vopt[i, j, k] = newV[0]

                            # Minh: sth wrong with this part. Try to avoid complex logic inside heterocl evaluation
                            # Evaluate convergence
                            # with hcl.if_(hcl.or_((newV - oldV) > epsilon[0], (oldV - newV) > epsilon[0])):
                            #    reSweep[0] = 1
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    # Note: Input to create_schedule should be hcl.placeholder type
    # These placeholder let the compiler knows size of input beforehand ---> faster execution
    Vinit = hcl.placeholder(tuple([10, 10, 10]), name="Vinit", dtype=hcl.Float())
    Vopt = hcl.placeholder(tuple([10, 10, 10]), name="Vopt", dtype=hcl.Float())
    gamma = hcl.placeholder((1,), "gamma")
    epsilon = hcl.placeholder((1,), "epsilon")
    reSweep = hcl.placeholder((1,), "reSweep")

    # Example of actions placeholder, rwd placeholder and transition placeholder
    actions = hcl.placeholder(tuple([10, 10, 10]), name="actions", dtype=hcl.Float())
    rwd     = hcl.placeholder(tuple([10, 10, 10]), name="rwd", dtype=hcl.Float())
    trans   = hcl.placeholder(tuple([10, 10, 10]), name="trans", dtype=hcl.Float())
    g_vs1   = hcl.placeholder((10, ), name="g_vs1", dtype=hcl.Float())
    g_vs2   = hcl.placeholder((10, ), name="g_vs2", dtype=hcl.Float())
    g_vs3   = hcl.placeholder((10, ), name="g_vs3", dtype=hcl.Float())

    # Create a static schedule -- graph
    s = hcl.create_schedule([Vinit, Vopt, actions, rwd, trans, gamma, epsilon, g_vs1, g_vs2, g_vs3], solve_Vopt)
    # Use this graph and build an executable
    f = hcl.build(s)

    # Now use this executable for our Python program

    # Convert the python array to hcl type array
    V_init   = hcl.asarray(np.zeros([10, 10, 10]))
    V_opt    = hcl.asarray(np.zeros([10, 10, 10]))
    actions  = hcl.asarray(np.zeros([10, 10, 10]))
    rwd      = hcl.asarray(np.zeros([10, 10, 10]))
    trans    = hcl.asarray(np.zeros([10, 10, 10]))
    gamma    = hcl.asarray(np.zeros(1))
    epsilon = hcl.asarray(np.zeros(1))
    g_vs_1 = hcl.asarray(np.linspace(-5.0, 5.0, 10))
    g_vs_2 = hcl.asarray(np.linspace(-5.0, 5.0, 10))
    g_vs_3 = hcl.asarray(np.linspace(-5.0, 5.0, 10))

    # Now use the executable
    f(V_init, V_opt, actions, rwd, trans, gamma, epsilon, g_vs_1, g_vs_2, g_vs_3)

    print(V_opt)
    print("I'm here\n")

# Test function
value_iteration_3D()