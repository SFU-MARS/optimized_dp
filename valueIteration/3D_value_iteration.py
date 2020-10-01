import heterocl as hcl
from Grid.GridProcessing import Grid
import numpy as np
import time
from user_definer import *



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
    Vopt = hcl.placeholder(tuple(g.pts_each_dim), name="Vopt", dtype=hcl.Float())
    discount = hcl.scalar(0, "gamma")
    epsilon  = hcl.scalar(0, "epsilon")
    reSweep = hcl.scalar(0, "reSweep")

    def solve_Vopt(Vinit, actions, rwd, trans, gamma, convergence_number):
        Vopt = hcl.asarray(Vinit)
        discount = gamma  
        epsilon  = convergence_number

        # Calculate the value of Vopt at point s given action a
        # TODO: Is it possible to make calls to an actions(s,a) or trans(s,a) func?
        def updateV(s):
            return max(rwd(s, a) + 
                discount*sum(p*Vopt[index_of_state(sNew)] for p, sNew in trans(s,a)) for a in actions)

        # Return the index values associated with the given stateValues
        # TODO: Find better way of obtaining the index from a set of state vals
        def index_of_state(stateVals, grid):
            i = round(stateVals[0] * grid.pts_each_dim[0] / (grid.max[0] - grid.min[0]))
            j = round(stateVals[1] * grid.pts_each_dim[1] / (grid.max[1] - grid.min[1]))
            k = round(stateVals[2] * grid.pts_each_dim[2] / (grid.max[2] - grid.min[2]))
            return (i, j, k)

        with hcl.Stage("Sweep_1"):
            reSweep = 1
            with hcl.while_(reSweep == 1):
                reSweep = 0
                with hcl.for_(0, g.pts_each_dim[0], name="i") as i:
                    with hcl.for_(0, g.pts_each_dim[1], name="j") as j:
                        with hcl.for_(0, g.pts_each_dim[2], name="k") as k:
                            s = hcl.placeholder((1,3), "state")
                            oldV = hcl.scalar(0, "oldV")
                            newV = hcl.scalar(0, "newV")
                            # Calculate the new optimal value
                            s = (g.vs[0][i,0,0], g.vs[1][0,j,0], g.vs[2][0,0,k])
                            oldV = Vopt[i, j, k]
                            newV = updateV(s)
                            Vopt[i, j, k] = newV[0]
                            
                            # Evaluate convergence
                            with hcl.if_(hcl.or_((newV - oldV) > epsilon, (oldV - newV) > epsilon)):
                                reSweep = 1
        return Vopt
    s = hcl.create_schedule([Vinit, actions, rwd, trans, gamma, convergence_number], solve_Vopt)
    return (hcl.build(s))

# Test function
value_iteration_3D()