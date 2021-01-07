import numpy as np
import math
import heterocl as hcl
import os



class MDP_3D_example:

    _bounds     = np.array([[-5.0, 5.0],[-5.0, 5.0],[-3.141592653589793, 3.141592653589793]])
    _ptsEachDim = np.array([25, 25, 9])
    _goal       = np.array([[3.5, 3.5], [1.5707, 2.3562]]) 

    # set _actions based on ranges and number of steps
    # format: range(lower bound, upper bound, number of steps)
    vValues  = np.linspace(-2, 2, 9)
    wValues  = np.linspace(-1, 1, 9)
    _actions = []
    for i in vValues:
        for j in wValues:
            _actions.append((i,j))
    _actions = np.array(_actions)

    _gamma    = np.array([0.93])
    _epsilon  = np.array([.3])
    _maxIters = np.array([500])
    _trans    = np.zeros([1, 4]) # size: [maximum number of transition states available x 4]
    _useNN    = np.array([0])
    _fillVal  = np.array([-400])

    # Given state and action, return successor states and their probabilities
    # sVals:  the coordinates of state
    # bounds: the lower and upper limits of the state space in each dimension
    # trans:  holds each successor state and the probability of reaching that state
    def transition(self, sVals, action, bounds, trans, goal):
        dx  = hcl.scalar(0, "dx")
        dy  = hcl.scalar(0, "dy")
        mag = hcl.scalar(0, "mag")

        # Check if moving from a goal state
        dx[0]  = sVals[0] - goal[0,0]
        dy[0]  = sVals[1] - goal[0,1]
        mag[0] = hcl.sqrt((dx[0] * dx[0]) + (dy[0] * dy[0]))
        with hcl.if_(hcl.and_(mag[0] <= 1.0, sVals[2] <= goal[1,1], sVals[2] >= goal[1,0])):
            trans[0, 0] = 0
        # Check if moving from an obstacle 
        with hcl.elif_(hcl.or_(sVals[0] < bounds[0,0] + 0.2, sVals[0] > bounds[0,1] - 0.2)):
            trans[0, 0] = 0
        with hcl.elif_(hcl.or_(sVals[1] < bounds[1,0] + 0.2, sVals[1] > bounds[1,1] - 0.2)):
            trans[0, 0] = 0
        # Standard move
        with hcl.else_():
            trans[0, 0] = 1.0
            trans[0, 1] = sVals[0] + (0.6 * action[0] * hcl.cos(sVals[2]))
            trans[0, 2] = sVals[1] + (0.6 * action[0] * hcl.sin(sVals[2]))
            trans[0, 3] = sVals[2] + (0.6 * action[1])
            # Adjust for periodic dimension
            with hcl.while_(trans[0, 3] > 3.141592653589793):
                trans[0, 3] -= 6.283185307179586
            with hcl.while_(trans[0, 3] < -3.141592653589793):
                trans[0, 3] += 6.283185307179586

    # Return the reward for taking action from state
    def reward(self, sVals, action, bounds, goal, trans):
        dx  = hcl.scalar(0, "dx")
        dy  = hcl.scalar(0, "dy")
        mag = hcl.scalar(0, "mag")
        rwd = hcl.scalar(0, "rwd")

        # Check if moving from a collision state, if so, assign a penalty
        with hcl.if_(hcl.or_(sVals[0] < bounds[0,0] + 0.2, sVals[0] > bounds[0,1] - 0.2)):
            rwd[0] = -400
        with hcl.elif_(hcl.or_(sVals[1] < bounds[1,0] + 0.2, sVals[1] > bounds[1,1] - 0.2)):
            rwd[0] = -400
        with hcl.else_():
            # Check if moving from a goal state
            dx[0]  = sVals[0] - goal[0,0]
            dy[0]  = sVals[1] - goal[0,1]
            mag[0] = hcl.sqrt((dx[0] * dx[0]) + (dy[0] * dy[0]))
            with hcl.if_(hcl.and_(mag[0] <= 1.0, sVals[2] <= goal[1,1], sVals[2] >= goal[1,0])):
                rwd[0] = 1000
            # Standard move
            with hcl.else_():
                rwd[0] = 0
        return rwd[0]
