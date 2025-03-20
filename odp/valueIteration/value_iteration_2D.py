import heterocl as hcl
import numpy as np

def updateVopt(obj, i, j, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, goal, ptsEachDim):
    p = hcl.scalar(0, "p")
    best_next_val = hcl.scalar(-1e9, "best_next_val")
    with hcl.for_(0, actions.shape[0], name="a") as a:
        # set iVals equal to (i,j,k) and sVals equal to the corresponding state values (si,sj,sk)
        updateStateVals(i, j, iVals, sVals, bounds, ptsEachDim)
        # call the transition function to obtain the outcome(s) of action a from state (si,sj,sk)
        obj.transition(sVals, iVals, actions[a], bounds, trans, goal)
        # initialize the value of the action using the immediate reward of taking that action
        intermeds[a] = obj.reward(sVals, iVals, actions[a], bounds, goal, trans)
        # add the value of each possible successor state to the estimated value of taking action a
        with hcl.for_(0, trans.shape[0], name="si") as si:
            p[0]     = trans[si,0]
            sVals[0] = trans[si,1]
            sVals[1] = trans[si,2]

            # convert the state values of the successor state (si,sj,sk) into indeces (ia,ij,ik)
            stateToIndex(sVals, iVals, bounds, ptsEachDim)
            intermeds[a] += (gamma[0] * (p[0] * Vopt[iVals[0], iVals[1]]))

        with hcl.if_(intermeds[a] > best_next_val[0]):
            best_next_val[0] = intermeds[a]

    Vopt[i, j] = best_next_val[0]


# convert state values into indeces using nearest neighbour
def stateToIndex(sVals, iVals, bounds, ptsEachDim):
    # iVals[0] = 0
    # iVals[1] = 0
    # with hcl.for_(0, sVals.shape[0], name="axis") as axis:
    with hcl.if_(sVals[0] <= bounds[0, 0]):
        iVals[0] = 0
    with hcl.elif_(sVals[0] >= bounds[0, 1]):
        iVals[0] = ptsEachDim[0] - 1
    with hcl.else_():
        iVals[0] = ((sVals[0] - bounds[0, 0]) / (bounds[0, 1] - bounds[0, 0])) * (ptsEachDim[0] - 1)
        iVals[0] = hcl.cast(hcl.Int(), iVals[0] + 0.5)

    with hcl.if_(sVals[1] <= bounds[1, 0]):
        iVals[1] = 0
    with hcl.elif_(sVals[1] >= bounds[1, 1]):
        iVals[1] = ptsEachDim[1] - 1
    with hcl.else_():
        iVals[1] = ((sVals[1] - bounds[1, 0]) / (bounds[1, 1] - bounds[1, 0])) * (ptsEachDim[1] - 1)
        iVals[1] = hcl.cast(hcl.Int(), iVals[1] + 0.5)

# convert indices into state values
def indexToState(iVals, sVals, bounds, ptsEachDim):
    sVals[0] = bounds[0,0] + ( (bounds[0,1] - bounds[0,0]) * (iVals[0] / (ptsEachDim[0]-1)) ) 
    sVals[1] = bounds[1,0] + ( (bounds[1,1] - bounds[1,0]) * (iVals[1] / (ptsEachDim[1]-1)) ) 


# set iVals equal to (i,j,k) and sVals equal to the corresponding state values at (i,j,k)
def updateStateVals(i, j, iVals, sVals, bounds, ptsEachDim):
    iVals[0] = i
    iVals[1] = j
    indexToState(iVals, sVals, bounds, ptsEachDim)


def value_iteration_2D(MDP_object):
    def solve_Vopt(Vopt, actions, intermeds, trans, gamma, iVals, sVals, bounds, goal, ptsEachDim):
            with hcl.Stage("Sweep_1"):
                with hcl.for_(0, Vopt.shape[0], name="i2") as i2:
                    with hcl.for_(0, Vopt.shape[1], name="j2") as j2:
                        updateVopt(MDP_object, i2, j2, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, goal, ptsEachDim)

            # Perform value iteration by sweeping in direction 1
            with hcl.Stage("Sweep_2"):
                with hcl.for_(0, Vopt.shape[0], name="i2") as i2:
                    with hcl.for_(0, Vopt.shape[1], name="j2") as j2:
                        i2 = Vopt.shape[0] - i2 - 1
                        j2 = Vopt.shape[1] - j2 - 1
                        updateVopt(MDP_object, i2, j2, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, goal, ptsEachDim)

            with hcl.Stage("Sweep_3"):
                with hcl.for_(0, Vopt.shape[0], name="i2") as i2:
                    with hcl.for_(0, Vopt.shape[1], name="j2") as j2:
                        j2 = Vopt.shape[1] - j2 - 1
                        updateVopt(MDP_object, i2, j2, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, goal, ptsEachDim)

            with hcl.Stage("Sweep_4"):
                with hcl.for_(0, Vopt.shape[0], name="i2") as i2:
                    with hcl.for_(0, Vopt.shape[1], name="j2") as j2:
                        i2 = Vopt.shape[0] - i2 - 1
                        j2 = Vopt.shape[1] - j2 - 1
                        updateVopt(MDP_object, i2, j2, iVals, sVals, actions, Vopt, intermeds, trans, gamma, bounds, goal, ptsEachDim)

    ###################################### SETUP PLACEHOLDERS ######################################
    

    # NOTE: trans is a tensor with size = maximum number of transitions
    # NOTE: intermeds must have size  [# possible actions]
    # NOTE: transition must have size [# possible outcomes, #state dimensions + 1]
    print()
    Vopt       = hcl.placeholder(tuple(MDP_object._ptsEachDim), name="Vopt", dtype=hcl.Float())
    gamma      = hcl.placeholder((0,), "gamma")
    actions    = hcl.placeholder(tuple(MDP_object._actions.shape), name="actions", dtype=hcl.Float())
    intermeds  = hcl.placeholder(tuple(MDP_object._actions.shape), name="intermeds", dtype=hcl.Float())
    trans      = hcl.placeholder(tuple(MDP_object._trans.shape), name="successors", dtype=hcl.Float())
    print(tuple(MDP_object._trans.shape))

    bounds     = hcl.placeholder(tuple(MDP_object._bounds.shape), name="bounds", dtype=hcl.Float())
    goal       = hcl.placeholder(tuple(MDP_object._goal.shape), name="goal", dtype=hcl.Float())
    ptsEachDim = hcl.placeholder(tuple([2]), name="ptsEachDim", dtype=hcl.Float())
    sVals      = hcl.placeholder(tuple([2]), name="sVals", dtype=hcl.Float())
    iVals      = hcl.placeholder(tuple([2]), name="iVals", dtype=hcl.Float())

    # Create a static schedule -- graph
    s = hcl.create_schedule([Vopt, actions, intermeds, trans, gamma, iVals, sVals, bounds, goal,
                             ptsEachDim], solve_Vopt)

    s_1 = solve_Vopt.Sweep_1
    s[s_1].parallel(s_1.i2)

    # Use this graph and build an executable
    return hcl.build(s) #target="llvm")