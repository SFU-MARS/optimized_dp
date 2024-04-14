import numpy as np
import math
import os
# import MDP.Example_3D as MDP
#import MDP.Example_6D as MDP
from odp.solver import *

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

class MDP_Pendulum_3D_example:
    _bounds = np.array([[-math.pi, math.pi], [-8., 8.], [0, 1]])
    _ptsEachDim = np.array([1000, 100, 1])
    # Set goal to be
    _goal = np.zeros([30, 30])
    a1 = np.linspace(-1., 1., _goal.shape[0], True)
    a2 = np.linspace(-1., 1., _goal.shape[1], True)

    # goals are actually angles
    for i in range(_goal.shape[0]):
        for j in range(_goal.shape[1]):
            _goal[i, j] = math.atan2(a2[j], a1[i])

    #print("goal")
    #print(np.sort(_goal.reshape(900,)))
    # set _actions based on ranges and number of steps
    torques = np.linspace(-2., 2., 100, endpoint=True)
    _actions = np.array(torques)

    _gamma = np.array([0.8])
    _epsilon = np.array([.00000005])
    _maxIters = np.array([200])
    # Deterministic case - dynamics based on pendulum dynamics
    _trans = np.zeros([1, 4])  # size: [maximum number of transition states available x 4]
    _useNN = np.array([1])
    _fillVal = np.array([-400])

    def __init__(self):
        # Some constant parameters for pendumlum from the openAI gym dynamics
        self.dt = 0.05
        self.g = 10
        self.m = 1.
        self.l = 1.
        self.max_speed = 8.
        self.coeff1 = 3 * self.g/ (2* self.l)
        self.coeff2 = 3.0/(self.m * self.l * self.l)

    # Given state and action, return successor states and their probabilities
    # sVals:  the coordinates of state
    # bounds: the lower and upper limits of the state space in each dimension
    # trans:  holds each successor state and the probability of reaching that state

    def arctan(self, x):
        my_st_result = hcl.scalar(0, "my_st_result")
        # Pay attention to the sign
        with hcl.if_(x <= 1):
            with hcl.if_(x >= -1):
                my_st_result[0] = x - x * x * x / 3 + x * x * x * x * x / 5 - x * x * x * x * x * x * x / 7 + x * x * x * x * x * x * x * x * x / 9 \
                         - x * x * x * x * x * x * x * x * x * x * x / 11 + x * x * x * x * x * x * x * x * x * x * x * x * x / 13 \
                         - x * x * x * x * x * x * x * x * x * x * x * x * x * x * x / 15 + x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x / 17 \
                         - x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x / 19 + x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x / 21 \
                         - x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x / 23 + x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x / 25

            with hcl.elif_(x < -1):
                my_st_result[0] = -math.pi / 2 - (1 / x - 1 / (x * x * x * 3) + 1 / (x * x * x * x * x * 5) - 1 / (
                            x * x * x * x * x * x * x * 7) + 1 / (x * x * x * x * x * x * x * x * x * 9) - 1 / (x * x * x * x * x * x * x * x * x * x * x * 11) + 1 / (
                                                              x * x * x * x * x * x * x * x * x * x * x * x * x * 13) \
                                                  - 1 / (
                                                              x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 15) + 1 / (
                                                              x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 17) \
                                                  - 1 / (
                                                              x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 19) + 1 / (
                                                              x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 21) \
                                                  - 1 / (
                                                              x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 23) + 1 / (
                                                              x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 25))
        with hcl.if_(x > 1):
            my_st_result[0] = math.pi / 2 - (1 / x - 1 / (x * x * x * 3) + 1 / (x * x * x * x * x * 5) - 1 / (
                        x * x * x * x * x * x * x * 7) + 1 / (x * x * x * x * x * x * x * x * x * 9) \
                                             - 1 / (x * x * x * x * x * x * x * x * x * x * x * 11) + 1 / (
                                                         x * x * x * x * x * x * x * x * x * x * x * x * x * 13) \
                                             - 1 / (
                                                         x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 15) + 1 / (
                                                         x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 17) \
                                             - 1 / (
                                                         x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 19) + 1 / (
                                                         x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 21) \
                                             - 1 / (
                                                         x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 23) + 1 / (
                                                         x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * 25))

        return my_st_result[0]


    def transition(self, sVals, iVals, action, bounds, trans, goal):
        # Variable declaration
        newthdot = hcl.scalar(0, "newthdot")
        th = hcl.scalar(0, "th")
        tan_th = hcl.scalar(0, "tan_th")
        new_th = hcl.scalar(0, "new_th")

        # Just use theta from goals variable
        # th[0] = goal[iVals[0], iVals[1]]
        th[0] = sVals[0]

        newthdot[0] = sVals[1] + (self.coeff1 * hcl.sin(th[0]) +  self.coeff2* action) * self.dt
        with hcl.if_(newthdot[0] > self.max_speed):
            newthdot[0] = self.max_speed
        with hcl.if_(newthdot[0] < -self.max_speed):
            newthdot[0] = -self.max_speed
        new_th[0] = th[0] + newthdot[0] * self.dt

        with hcl.if_(newthdot[0] > math.pi):
            new_th[0] = new_th[0] - 2*math.pi
        with hcl.if_(newthdot[0] < -math.pi):
            new_th[0] = new_th[0] + 2*math.pi
        trans[0, 0] = 1.0
        trans[0, 1] = new_th[0]
        trans[0, 2] = newthdot[0]
        trans[0, 3] = sVals[2]

    # Return the reward for taking action from state
    def reward(self, sVals, iVals, action, bounds, goal, trans):

        # Variable declaration
        th = hcl.scalar(0, "th")
        tan_th = hcl.scalar(0, "tan_th")
        rwd = hcl.scalar(0, "rwd")
        # Infer theta from x,y
        # tan_th[0] = sVals[1] / sVals[0]
        th[0] = sVals[0]
        # rwd[0] = -(th[0] * th[0] + 0.1 * sVals[2] * sVals[2] + 0.001 * action * action)
        rwd[0] = -(th[0] * th[0] + 0.1 * sVals[1] * sVals[1] + 0.001 * action * action)
        # rwd[0] = -(th[0] * th[0])
        return rwd[0]

    # Provide a print function
    def writeResults(self, V, dir_path, file_name, just_values=False):
        # Create directory for results if one does not exist
        print("\nRecording results")
        try:
            os.mkdir(dir_path)
            print("Created directory: ", dir_path)
        except:
            print("Writing to: '", dir_path, "'")
        # Open file and write results
        f = open(dir_path + file_name, "w")
        for k in range(V.shape[2]):
            for i in range(V.shape[0]):
                for j in range(V.shape[1]):
                    s = ""
                    if not just_values:
                        si = ((i / (self._ptsEachDim[0] - 1)) * (self._bounds[0, 1] - self._bounds[0, 0])) + self._bounds[0, 0]
                        sj = ((j / (self._ptsEachDim[1] - 1)) * (self._bounds[1, 1] - self._bounds[1, 0])) + self._bounds[1, 0]
                        sk = ((k / (self._ptsEachDim[2] - 1)) * (self._bounds[2, 1] - self._bounds[2, 0])) + self._bounds[2, 0]
                        state = ("{:.4f}".format(si), "{:.4f}".format(sj), "{:.4f}".format(sk))
                        s = str(state) + "   " + str("{:.4f}".format(V[(i, j, k)])) + '\n'
                    else:
                        s = str("{:.4f}".format(V[(i, j, k)])) + ',\n'
                    f.write(s)
        print("Finished recording results")

myProblem   = MDP_Pendulum_3D_example()
result = solveValueIteration(myProblem)
print(np.sort(result.reshape(100000))[-10:])
print(np.max(result))
print(result.shape)
# myProblem.writeResults(result, )
np.save('pendulum.npy', result.reshape((1000, 100)))

# print(result)
# Optionally provide a directory and filename to save results of computation
# dir_path   = None
# file_name  = None

