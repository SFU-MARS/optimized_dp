import numpy as np
import math
import os


_bounds     = np.array([[-2.5, 2.5],[-2.5, 2.5],[-2.5, 2.5],[-2.5, 2.5],[-2.5, 2.5],[-2.5, 2.5]])
_ptsEachDim = np.array([5, 5, 5, 5, 5, 5])
_goal       = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5]) 

# set _actions based on ranges and number of steps
# format: range(lower bound, upper bound, number of steps)
iValues  = np.linspace(-0.5, 0.5, 3)
jValues  = np.linspace(-0.5, 0.5, 3)
kValues  = np.linspace(-0.5, 0.5, 3)
lValues  = np.linspace(-0.5, 0.5, 3)
mValues  = np.linspace(-0.5, 0.5, 3)
nValues  = np.linspace(-0.5, 0.5, 3)
_actions = []
for i in iValues:
    for j in jValues:
        for k in kValues:
            for l in lValues:
                for m in mValues:
                    for n in nValues:
                        _actions.append((i,j,k,l,m,n))
_actions = np.array(_actions)

_gamma      = np.array([0.93])
_epsilon    = np.array([.3])
_maxIters   = np.array([30])
_trans      = np.zeros([1, 7]) # size: [maximum number of transition states available x 4]
_useNN      = np.array([1])

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
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            for k in range(V.shape[2]):
                for l in range(V.shape[3]):
                    for m in range(V.shape[4]):
                        for n in range(V.shape[5]):
                            s = ""
                            if not just_values:
                                si = (( i / (_ptsEachDim[0] - 1) ) * (_bounds[0,1] - _bounds[0,0])) + _bounds[0,0]
                                sj = (( j / (_ptsEachDim[1] - 1) ) * (_bounds[1,1] - _bounds[1,0])) + _bounds[1,0]
                                sk = (( k / (_ptsEachDim[2] - 1) ) * (_bounds[2,1] - _bounds[2,0])) + _bounds[2,0]
                                sl = (( l / (_ptsEachDim[3] - 1) ) * (_bounds[3,1] - _bounds[3,0])) + _bounds[3,0]
                                sm = (( m / (_ptsEachDim[4] - 1) ) * (_bounds[4,1] - _bounds[4,0])) + _bounds[4,0]
                                sn = (( n / (_ptsEachDim[5] - 1) ) * (_bounds[5,1] - _bounds[5,0])) + _bounds[5,0]
                                
                                state = ("{:.4f}".format(si), "{:.4f}".format(sj), "{:.4f}".format(sk), 
                                         "{:.4f}".format(sl), "{:.4f}".format(sm), "{:.4f}".format(sn))
                                s = str(state) + "   " + str("{:.4f}".format(V[(i,j,k,l,m,n)])) + '\n'
                            else:
                                s = str("{:.4f}".format(V[(i,j,k,l,m,n)])) + ',\n'
                            f.write(s)
    print("Finished recording results")