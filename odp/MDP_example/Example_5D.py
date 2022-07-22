import numpy as np
import heterocl as hcl
import os



class MDP_5D_example:

    _bounds     = np.array([[-2.5, 2.5],[-2.5, 2.5],[-2.5, 2.5],[-2.5, 2.5],[-2.5, 2.5]])
    _ptsEachDim = np.array([10, 10, 10, 10, 10])
    _goal       = np.array([1.5, 1.5, 1.5, 1.5, 1.5]) 

    # set _actions based on ranges and number of steps
    # format: range(lower bound, upper bound, number of steps)
    iValues  = np.linspace(-0.5, 0.5, 3)
    jValues  = np.linspace(-0.5, 0.5, 3)
    kValues  = np.linspace(-0.5, 0.5, 3)
    lValues  = np.linspace(-0.5, 0.5, 3)
    mValues  = np.linspace(-0.5, 0.5, 3)
    _actions = []
    for i in iValues:
        for j in jValues:
            for k in kValues:
                for l in lValues:
                    for m in mValues:
                        _actions.append((i,j,k,l,m))
    _actions = np.array(_actions)

    _gamma      = np.array([0.93])
    _epsilon    = np.array([.3])
    _maxIters   = np.array([30])
    _trans      = np.zeros([1, 7]) # size: [maximum number of transition states available x 4]
    _useNN      = np.array([1])


    # Given state and action, return successor states and their probabilities
    # sVals:  the coordinates of state
    # bounds: the lower and upper limits of the state space in each dimension
    # trans:  holds each successor state and the probability of reaching that state
    def transition(self, sVals, action, bounds, trans, goal):
        di  = hcl.scalar(0, "di") 
        dj  = hcl.scalar(0, "dj") 
        dk  = hcl.scalar(0, "dk") 
        dl  = hcl.scalar(0, "dl") 
        dm  = hcl.scalar(0, "dm") 
        mag = hcl.scalar(0, "mag")

        # Check if moving from a goal state
        di[0]  = (sVals[0] - goal[0]) * (sVals[0] - goal[0])
        dj[0]  = (sVals[1] - goal[1]) * (sVals[1] - goal[1])
        dk[0]  = (sVals[2] - goal[2]) * (sVals[2] - goal[2])
        dl[0]  = (sVals[3] - goal[3]) * (sVals[3] - goal[3])
        dm[0]  = (sVals[4] - goal[4]) * (sVals[4] - goal[4])
        mag[0] = hcl.sqrt(di[0] + dj[0] + dk[0] + dl[0] + dm[0])

        # Check if moving from an obstacle 
        with hcl.if_(mag[0] <= 5.0):
            trans[0, 0] = 0
        with hcl.elif_(hcl.or_(sVals[0] <= bounds[0,0], sVals[0] >= bounds[0,1])):
            trans[0, 0] = 0
        with hcl.elif_(hcl.or_(sVals[1] <= bounds[1,0], sVals[1] >= bounds[1,1])):
            trans[0, 0] = 0
        with hcl.elif_(hcl.or_(sVals[2] <= bounds[2,0], sVals[2] >= bounds[2,1])):
            trans[0, 0] = 0
        with hcl.elif_(hcl.or_(sVals[3] <= bounds[3,0], sVals[3] >= bounds[3,1])):
            trans[0, 0] = 0
        with hcl.elif_(hcl.or_(sVals[4] <= bounds[4,0], sVals[4] >= bounds[4,1])):
            trans[0, 0] = 0

        # Standard move
        with hcl.else_():
            trans[0, 0] = 1.0
            trans[0, 1] = sVals[0] + action[0]
            trans[0, 2] = sVals[1] + action[1]
            trans[0, 3] = sVals[2] + action[2]
            trans[0, 4] = sVals[3] + action[3]  
            trans[0, 5] = sVals[4] + action[4]  

    # Return the reward for taking action from state
    def reward(self, sVals, action, bounds, goal, trans):
        di  = hcl.scalar(0, "di") 
        dj  = hcl.scalar(0, "dj") 
        dk  = hcl.scalar(0, "dk") 
        dl  = hcl.scalar(0, "dl") 
        dm  = hcl.scalar(0, "dm") 
        mag = hcl.scalar(0, "mag")
        rwd = hcl.scalar(0, "rwd")

        # Check if moving from a collision state, if so, assign a penalty
        with hcl.if_(hcl.or_(sVals[0] <= bounds[0,0], sVals[0] >= bounds[0,1])):
            rwd[0] = -400
        with hcl.elif_(hcl.or_(sVals[1] <= bounds[1,0], sVals[1] >= bounds[1,1])):
            rwd[0] = -400
        with hcl.elif_(hcl.or_(sVals[2] <= bounds[2,0], sVals[2] >= bounds[2,1])):
            rwd[0] = -400
        with hcl.elif_(hcl.or_(sVals[3] <= bounds[3,0], sVals[3] >= bounds[3,1])):
            rwd[0] = -400
        with hcl.elif_(hcl.or_(sVals[4] <= bounds[4,0], sVals[4] >= bounds[4,1])):
            rwd[0] = -400

        with hcl.else_():
            # Check if moving from a goal state
            di[0]  = (sVals[0] - goal[0]) * (sVals[0] - goal[0])
            dj[0]  = (sVals[1] - goal[1]) * (sVals[1] - goal[1])
            dk[0]  = (sVals[2] - goal[2]) * (sVals[2] - goal[2])
            dl[0]  = (sVals[3] - goal[3]) * (sVals[3] - goal[3])
            dm[0]  = (sVals[4] - goal[4]) * (sVals[4] - goal[4])
            mag[0] = hcl.sqrt(di[0] + dj[0] + dk[0] + dl[0] + dm[0])
            with hcl.if_(mag[0] <= 5.0):
                rwd[0] = 1000
            # Standard move
            with hcl.else_():                
                rwd[0] = 0
        return rwd[0]

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
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                for k in range(V.shape[2]):
                    for l in range(V.shape[3]):
                        for m in range(V.shape[4]):
                            s = ""
                            if not just_values:
                                si = ((i / (self._ptsEachDim[0] - 1)) * (self._bounds[0, 1] - self._bounds[0, 0])) + self._bounds[0, 0]
                                sj = ((j / (self._ptsEachDim[1] - 1)) * (self._bounds[1, 1] - self._bounds[1, 0])) + self._bounds[1, 0]
                                sk = ((k / (self._ptsEachDim[2] - 1)) * (self._bounds[2, 1] - self._bounds[2, 0])) + self._bounds[2, 0]
                                sl = ((l / (self._ptsEachDim[3] - 1)) * (self._bounds[3, 1] - self._bounds[3, 0])) + self._bounds[3, 0]
                                sm = ((m / (self._ptsEachDim[4] - 1)) * (self._bounds[4, 1] - self._bounds[4, 0])) + self._bounds[4, 0]

                                state = ("{:.4f}".format(si), "{:.4f}".format(sj), "{:.4f}".format(sk),
                                         "{:.4f}".format(sl), "{:.4f}".format(sm))
                                s = str(state) + "   " + str("{:.4f}".format(V[(i, j, k, l, m)])) + '\n'
                            else:
                                s = str("{:.4f}".format(V[(i, j, k, l, m)])) + ',\n'
                            f.write(s)
        print("Finished recording results")
