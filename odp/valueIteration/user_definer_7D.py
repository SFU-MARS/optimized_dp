import numpy as np
import os
import MDP.Example_7D as MDP
import valueIteration as VI


myProblem   = MDP.MDP_7D_example()

_bounds     = myProblem._bounds
_ptsEachDim = myProblem._ptsEachDim
_goal       = myProblem._goal
_actions    = myProblem._actions
_gamma      = myProblem._gamma    
_epsilon    = myProblem._epsilon  
_maxIters   = myProblem._maxIters 
_trans      = myProblem._trans    
_useNN      = myProblem._useNN    

transition = myProblem.transition
reward     = myProblem.reward


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
                            for o in range(V.shape[6]):
                                s = ""
                                if not just_values:
                                    si = (( i / (_ptsEachDim[0] - 1) ) * (_bounds[0,1] - _bounds[0,0])) + _bounds[0,0]
                                    sj = (( j / (_ptsEachDim[1] - 1) ) * (_bounds[1,1] - _bounds[1,0])) + _bounds[1,0]
                                    sk = (( k / (_ptsEachDim[2] - 1) ) * (_bounds[2,1] - _bounds[2,0])) + _bounds[2,0]
                                    sl = (( l / (_ptsEachDim[3] - 1) ) * (_bounds[3,1] - _bounds[3,0])) + _bounds[3,0]
                                    sm = (( m / (_ptsEachDim[4] - 1) ) * (_bounds[4,1] - _bounds[4,0])) + _bounds[4,0]
                                    sn = (( n / (_ptsEachDim[5] - 1) ) * (_bounds[5,1] - _bounds[5,0])) + _bounds[5,0]
                                    so = (( o / (_ptsEachDim[6] - 1) ) * (_bounds[6,1] - _bounds[6,0])) + _bounds[6,0]
                                    
                                    state = ("{:.4f}".format(si), "{:.4f}".format(sj), "{:.4f}".format(sk), 
                                            "{:.4f}".format(sl), "{:.4f}".format(sm), "{:.4f}".format(sn), "{:.4f}".format(so))
                                    s = str(state) + "   " + str("{:.4f}".format(V[(i,j,k,l,m,n,o)])) + '\n'
                                else:
                                    s = str("{:.4f}".format(V[(i,j,k,l,m,n,o)])) + ',\n'
                                f.write(s)
    print("Finished recording results")
    