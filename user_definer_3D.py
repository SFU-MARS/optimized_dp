import numpy as np
import math
import os
import MDP.Example_3D as MDP
import valueIteration as VI



####################################################################################################
#                                                                                                  #
#                                  VALUE ITERATION DOCUMENTATION                                   #
#                                                                                                  #
#                                                                                                  #
# 1. REQUIRED FUNCTIONS                                                                            #
# The user is required to provide implementations of the following state transition and reward     #
# functions within ./MDP/:                                                                         #
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

myProblem   = MDP.MDP_3D_example()

_bounds     = myProblem._bounds
_ptsEachDim = myProblem._ptsEachDim
_goal       = myProblem._goal
_actions    = myProblem._actions
_gamma      = myProblem._gamma    
_epsilon    = myProblem._epsilon  
_maxIters   = myProblem._maxIters 
_trans      = myProblem._trans    
_useNN      = myProblem._useNN    
_fillVal    = myProblem._fillVal  

transition = myProblem.transition
reward     = myProblem.reward

# Optionally provide a directory and filename to save results of computation
dir_path   = None
file_name  = None

# Provide a print function 
def writeResults(V, dir_path, file_name, just_values=False):
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
                    si = (( i / (_ptsEachDim[0] - 1) ) * (_bounds[0,1] - _bounds[0,0])) + _bounds[0,0]
                    sj = (( j / (_ptsEachDim[1] - 1) ) * (_bounds[1,1] - _bounds[1,0])) + _bounds[1,0]
                    sk = (( k / (_ptsEachDim[2] - 1) ) * (_bounds[2,1] - _bounds[2,0])) + _bounds[2,0]
                    state = ("{:.4f}".format(si), "{:.4f}".format(sj), "{:.4f}".format(sk))
                    s = str(state) + "   " + str("{:.4f}".format(V[(i,j,k)])) + '\n'
                else:
                    s = str("{:.4f}".format(V[(i,j,k)])) + ',\n'
                f.write(s)
    print("Finished recording results")