import numpy as np
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