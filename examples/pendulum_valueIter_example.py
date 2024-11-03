import numpy as np
import math
import os
# import MDP.Example_3D as MDP
#import MDP.Example_6D as MDP
from odp.solver import *
# from odp.valueIteration import value_iteration_2D

import numpy as np
import math
import heterocl as hcl
import os

####################################################################################################
#                                                                                                  #
#                                  VALUE ITERATION DOCUMENTATION                                   #
#                                                                                                  #
#                                                                                                  #
# 1. REQUIRED FUNCTIONS                                                                            #
# The user is required to provide implementations of the following state transition and reward     #
# functions:                                                                                       #
#                                                                                                  #
#       transition(sVals, action, bounds, trans, goal)                                             #
#       reward(sVals, action, bounds, goal, trans)                                                 #
#                                                                                                  #
#       1.1 PARAMETERS                                                                             #
#       sVals:  a vector containing state values                                                   #
#               Format: sVals[0] = state dimension 1, sVals[1] = state dimension 2, .....          #
#       action: the action taken from the state specified by sVals                                 #
#               Format: user-specified. This is used exclusively in user-defined functions         #
#       bounds: the lower and upper limits of the state space in each dimension                    #
#               Format: bounds[i,0] is the lower bound of dimension i, bounds[i,1] is the upper    #
#       trans:  a matrix containing the possible successor states when taking action from the      #
#               state defined by sVals.                                                            #
#               Format: the size of this vector must be:                                           #
#                       maximum number of possible transitions x number of state dimensions + 1    #
#                       The first element of each row of trans will be a probability, and the      #
#                       remaining elements will be the state values of that transition state       #
#                       ex. trans[i, 0] is the probability of making transition i,                 #
#                           trans[i, 1] is the first state dimension of transition i               #
#       goal:   a vector defining the goal region                                                  #
#               Format: user-specified. This is used exclusively in user-defined functions         #
#                                                                                                  #
# 2. PROBLEM DECLARATION                                                                           #
# The user is required to provide declarations of a series of variables that define the value      #
# iteration problem:                                                                               #
#                                                                                                  #
#       2.1 VARIABLE DECLARATIONS                                                                  #
#       _bounds:     a matrix defining the state space boundaries                                  #
#                    Format: np.array([[iMin, iMax], [jMin, jMax], ... , [nMin, nMax])             #
#       _ptsEachDim: the number of grid spaces for each dimension                                  #
#                    Format: np.array([ptsDimension1, ptsDimension2, ... , ptsDimensionN])         #
#       _actions:    The set of actions that can be taken from any state                           #
#                    Format: np.array([action1, action2, ... , actionN])                           #
#       _gamma:      the discount factor                                                           #
#                    Format: np.array([value])                                                     #
#       _epsilon:    the convergence number. The algorithm is considered to have converged to the  #
#                    optimal value function when the largest difference between values is less     #
#                    than _epsilon.                                                                #
#                    Format: np.array([value])                                                     #
#       _maxIters:   the maximum number of iterations that will be performed without convergence   #
#                    Format: np.array([value])                                                     #
#       _trans:      a placeholder matrix containing possible successor states.                    #
#                    Format: np.zeros([maximumNumberOfTransitionStates, 1 + numberOfDimensions])   #
#                    Usage:  The first value of each row is the probability of the transition. The #
#                            remaining n values of that row are the n values of the state space    #
#       _useNN:      a flag for choosing between Nearest Neighbour and Linear Interpolation modes  #
#                    Format: np.array([0]) to use Linear Interpolation                             #
#                            np.array([1]) to use Nearest Neighbour                                #
#       _fillVal     the fill value for the Linear Interpolation method. This value is assigned to #
#                    states that are out of the state space bounds.                                #
#                    Format: np.array([value])                                                     #
#                    Usage: this variable must be declared, even if using the Nearest Neighbour    #
#                           method. In this case, the value assigned is of no significance.        #
#                                                                                                  #
#                                                                                                  #
#                                                                                                  #
####################################################################################################

class pendulum_2d_example:
    _bounds = np.array([[-math.pi, math.pi], [-8., 8.]])
    _ptsEachDim = np.array([201, 401])
    # Set goal to b
    _goal = np.zeros([30, 30])
    torques = np.linspace(-2., 2., 41)
    _actions = torques #np.array(torques)

    _gamma = np.array([0.99])
    _epsilon = np.array([1.117e-5])
    _maxIters = np.array([1500])
    # Deterministic case - dynamics based on pendulum dynamics
    _trans = np.zeros([1, 3])  # size: [maximum number of transition states available x 4]

    def __init__(self):
        # Some constant parameters for pendumlum from the openAI gym dynamics
        self.dt = 0.05
        self.g = 10
        self.m = 1.
        self.l = 1.
        self.max_speed = 8.
        self.coeff1 = 3 * self.g/ (2* self.l)
        self.coeff2 = 3.0/(self.m * self.l * self.l)
        print("fick")

    def transition(self, sVals, iVals, u, bounds, trans, goal):
        # Variable declaration
        newthdot = hcl.scalar(0, "newthdot")
        th = hcl.scalar(0, "th")
        new_th = hcl.scalar(0, "new_th")


        # Just use theta from goals variable
        # th[0] = goal[iVals[0], iVals[1]]
        th[0] = sVals[0]

        newthdot[0] = sVals[1] + (self.coeff1 * hcl.sin(sVals[0]) +  self.coeff2 * u) * self.dt
        with hcl.if_(newthdot[0] > self.max_speed):
            newthdot[0] = self.max_speed
        with hcl.if_(newthdot[0] < -self.max_speed):
            newthdot[0] = -self.max_speed
        new_th[0] = th[0] + newthdot[0] * self.dt

        # Normalize angles
        with hcl.if_(new_th[0] >= math.pi):
            new_th[0] = new_th[0] - 2*math.pi
        with hcl.elif_(new_th[0] < -math.pi):
            new_th[0] = new_th[0] + 2*math.pi
        trans[0, 0] = 1.0
        trans[0, 1] = new_th[0]
        trans[0, 2] = newthdot[0]

    # Return the reward for taking action from state
    def reward(self, sVals, iVals, u, bounds, goal, trans):
        # Variable declaration
        rwd = hcl.scalar(0, "rwd")
        rwd[0] = -(sVals[0] * sVals[0] + 0.1 * sVals[1] * sVals[1] + 0.001 * u * u)
        return rwd[0]


myProblem   = pendulum_2d_example()
print(myProblem)

result = solveValueIteration(myProblem)
print(np.shape(result))
# print(result[20, 25])
print(np.max(result))
print(np.min(result))
print(result)
np.save('hcl_pendulum_res_new.npy', result)


# myProblem.writeResults(result, )
# python_res = np.load('pendulum_python_gamma08_new.npy')
V = np.load("pendulum_python_gamma99_new.npy")

print(V)
# print(result)
# Optionally provide a directory and filename to save results of computation
# dir_path   = None
# file_name  = None

