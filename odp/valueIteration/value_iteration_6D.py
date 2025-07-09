import heterocl as hcl
import numpy as np

def updateVopt(obj, current_indices, actions, V, gamma, bounds, ptsEachDim):
    p = hcl.scalar(0, "p")
    best_next_val = hcl.scalar(-1e9, "best_next_val")
    intermeds = hcl.compute((actions.shape[0], ), lambda *x: 0, "intermeds")
    with hcl.for_(0, actions.shape[0], name="a") as a:

        sVals = discreteIndexToContinuousState(current_indices, bounds, ptsEachDim)
        # call the transition function to obtain the outcome(s) of action a from state (si,sj,sk)
        transition_matrix = obj.transition(sVals, current_indices, actions[a])
        # initialize the value of the action using the immediate reward of taking that action
        intermeds[a] = obj.reward(sVals, current_indices, actions[a])
        # add the value of each possible successor state to the estimated value of taking action a
        with hcl.for_(0, transition_matrix.shape[0], name="si") as si:
            p[0]     = transition_matrix[si,0]
            sVals[0] = transition_matrix[si,1]
            sVals[1] = transition_matrix[si,2]
            sVals[2] = transition_matrix[si,3]
            sVals[3] = transition_matrix[si,4]
            sVals[4] = transition_matrix[si,5]
            sVals[5] = transition_matrix[si,6]

            # convert the state values of the successor state (si,sj,sk) into indeces (ia,ij,ik)
            iVals = continuousStateToDiscreteIndices(sVals, bounds, ptsEachDim)
            intermeds[a] += (gamma[0] * (p[0] * V[iVals[0], iVals[1], iVals[2], iVals[3], iVals[4], iVals[5]]))

        with hcl.if_(intermeds[a] > best_next_val[0]):
            best_next_val[0] = intermeds[a]

    return best_next_val[0]


# convert state values into indeces using nearest neighbour
def continuousStateToDiscreteIndices(sVals, bounds, ptsEachDim):
    iVals = hcl.compute((6,), lambda *x: 0, "iVals") 
   
    for axis in range(6):
        with hcl.if_(sVals[axis] <= bounds[axis, 0]):
            iVals[axis] = 0
        with hcl.elif_(sVals[axis] >= bounds[axis, 1]):
            iVals[axis] = ptsEachDim[axis] - 1
        with hcl.else_():
            iVals[axis] = ((sVals[axis] - bounds[axis, 0]) / (bounds[axis, 1] - bounds[axis, 0])) * (ptsEachDim[axis] - 1)
            iVals[axis] = hcl.cast(hcl.Int(), iVals[axis] + 0.5)

    return iVals

# convert indices into state valuestrans
def discreteIndexToContinuousState(iVals, bounds, ptsEachDim):
    sVals = hcl.compute((6,), lambda *x: 0, "sVals")
    for axis in range(6):
        sVals[axis] = bounds[axis, 0] + ( (bounds[axis, 1] - bounds[axis, 0]) * (iVals[axis] / (ptsEachDim[axis]-1)))
    return sVals

def value_iteration_6D(MDP_object, pts_each_dim, bounds, 
                       action_space):
    def solve_Vopt(Vopt, actions, gamma, bounds, ptsEachDim):
            for sweep_id in range(5):
                with hcl.Stage(f"Sweep_{sweep_id + 1}"):
                    with hcl.for_(0, Vopt.shape[0], name="i") as i:
                        with hcl.for_(0, Vopt.shape[1], name="j") as j:
                            with hcl.for_(0, Vopt.shape[2], name="k") as k:
                                with hcl.for_(0, Vopt.shape[3], name="l") as l:
                                    with hcl.for_(0, Vopt.shape[4], name="m") as m:
                                        with hcl.for_(0, Vopt.shape[5], name="n") as n:
                                            if sweep_id & 1:
                                                i = Vopt.shape[0] - i - 1
                                            if sweep_id & 2:
                                                j = Vopt.shape[1] - j - 1
                                            if sweep_id & 4:
                                                k = Vopt.shape[2] - k - 1
                                            if sweep_id & 8:
                                                l = Vopt.shape[3] - l - 1
                                            if sweep_id & 16:
                                                m = Vopt.shape[4] - m - 1
                                            if sweep_id & 32:
                                                n = Vopt.shape[5] - n - 1
                                            indices = (i, j, k, l, m, n)
                                            Vopt[i, j, k, l, m, n] = updateVopt(MDP_object, indices, actions, Vopt, 
                                                                    gamma, bounds, ptsEachDim)


    ###################################### SETUP PLACEHOLDERS ######################################
    

    # NOTE: trans is a tensor with size = maximum number of transitions
    # NOTE: intermeds must have size  [# possible actions]
    # NOTE: transition must have size [# possible outcomes, #state dimensions + 1]
    Vopt       = hcl.placeholder(tuple(pts_each_dim), name="Vopt", dtype=hcl.Float())
    gamma      = hcl.placeholder((0,), "gamma")
    actions    = hcl.placeholder(tuple(action_space.shape), name="actions", dtype=hcl.Float())

    bounds     = hcl.placeholder(tuple(bounds.shape), name="bounds", dtype=hcl.Float())
    ptsEachDim = hcl.placeholder((6,), name="ptsEachDim", dtype=hcl.Float())

    # Create a static schedule -- graph
    s = hcl.create_schedule([Vopt, actions, gamma, bounds, ptsEachDim],
                             solve_Vopt)

    for sweep_id in range(5):
        sweep = getattr(solve_Vopt, "Sweep_{}".format(sweep_id + 1))
        s[sweep].parallel(sweep.i) 

    # Use this graph and build an executable
    return hcl.build(s)