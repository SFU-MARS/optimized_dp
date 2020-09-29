import heterocl as hcl
from Grid.GridProcessing import Grid
import numpy as np
import time


# Initialize the value function as a zeros array
# dimensions: number of indeces along each axis of the stateSpace
def initialize(dimensions):
    with hcl.Stage("Value Iteration"):
        Vinit = np.zeros(dimensions)
        return hcl.asarray(Vinit)


# Perform value iteration to solve for the optimal value function, Vopt
# grid:       a Grid object characterizing the state space
# actions:    the set of actions a that can be taken from any state 
#             TODO: extend for state-specific actions
# rwd:        the transition reward function
# trans:      given state s, action a, return the set of successor states s' and
#             the associated probability p of reaching s'
# gamma:      discount factor
# epsilon:    a small number
def valueIteration3D(grid, actions, rwd, trans, gamma, epsilon):
    Vopt = initialize(grid.pts_each_dim)
    Vopt = sweep(Vopt, grid, actions, rwd, trans, gamma, epsilon)
    return Vopt


# Perform value iteration by sweeping along each possible diagonal
# TODO: implement all possible diagonal sweeps
def sweep(Vopt, grid, actions, rwd, trans, gamma, epsilon):
    while (sweepOne(Vopt, grid, actions, rwd, trans, gamma)): continue


# Sweep direction 1
def sweepOne(Vopt, grid, actions, rwd, trans, gamma, epsilon):
    reSweep    = hcl.scalar(0, "reSweep")
    reSweep[0] = 0
    with hcl.for_(0, grid.pts_each_dim[0], name="i") as i:
        with hcl.for_(0, grid.pts_each_dim[1], name="j") as j:
            with hcl.for_(0, grid.pts_each_dim[2], name="k") as k:
                s    = hcl.scalar((1,3), "state")
                oldV = hcl.scalar(0, "oldV")
                newV = hcl.scalar(0, "newV")
                # Calculate the new optimal value
                s[:]    = (grid.vs[0][(i,0,0)], grid.vs[1][(0,j,0)], grid.vs[2][(0,0,k)]) 
                oldV[0] = Vopt[i, j, k]
                newV[0] = updateV(Vopt, s, a, actions, rwd, trans, gamma)
                Vopt[i, j, k] = newV[0]
                with hcl.if_(abs(newV[0] - oldV[0]) > epsilon): reSweep[0] = 1
    return reSweep


# Calculate the value of Vopt at point s given action a
def updateV(Vopt, s, a, actions, rwd, trans, gamma):
    return max(rwd(s, a) + 
        (gamma * sum(p * Vopt(indexGivenState(sNew)) for p, sNew in trans(s, a))) for a in actions) 


# Return the state values associated with the integer indeces of Vopt
def calcStateValues(coordinate, grid):
    Si = (coordinate[0] / grid.pts_each_dim[0]) * (grid.max[0] - grid.min[0])
    Sj = (coordinate[1] / grid.pts_each_dim[1]) * (grid.max[1] - grid.min[1])
    Sk = (coordinate[2] / grid.pts_each_dim[2]) * (grid.max[2] - grid.min[2])
    return (Si, Sj, Sk)


# Return the index values associated with the given stateValues
def indexGivenState(stateVals, grid):
    i = round(stateVals[0] * grid.pts_each_dim[0] / (grid.max[0] - grid.min[0]))
    j = round(stateVals[1] * grid.pts_each_dim[1] / (grid.max[1] - grid.min[1]))
    k = round(stateVals[2] * grid.pts_each_dim[2] / (grid.max[2] - grid.min[2]))
    return (i, j, k)
