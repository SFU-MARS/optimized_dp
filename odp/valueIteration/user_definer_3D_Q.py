import numpy as np
import os

_bounds     = np.array([[-5.0, 5.0],[-5.0, 5.0],[-3.1415, 3.1415]])
_ptsEachDim = np.array([25, 25, 9, 81])
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
_fillVal  = np.array([-400]) # Fill Value for linear interpolation

# Write results to file
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
                    s = str(state) + "   " + str(max(V[(i,j,k)])) + '\n'
                else:
                    s = str("{:.4f}".format(max(V[(i,j,k)]))) + ',\n'
                f.write(s)
    print("Finished recording results")
