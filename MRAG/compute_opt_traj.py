from src.odp.GridProcessing import Grid
from src.odp.first_order_generic import spa_deriv
from src.odp.update_state import next_state, update_state
import numpy as np
from src.odp.find_tEarlist import *

def compute_opt_traj(grid: Grid, V, tau, dyn_sys, subsamples=1, arriveAfter = None, obstVal = None):
    """
    Computes the optimal trajectory, controls and disturbance to a minimal BRT/BRS

    Args:
        grid:
        V:
        tau:
        dyn_sys:
        subsamples: Number of times opt_u and opt_d are calculated within dt

    Returns:
        traj: State of dyn_sys from time tau[-1] to tau[0]
        opt_u: Optimal control at each time
        opt_d: Optimal disturbance at each time

    """
    assert V.shape[-1] == len(tau)

    dt = (tau[1] - tau[0]) / subsamples

    # first entry is dyn_sys at time tau[-1]
    # second entry is dyn_sys at time tau[-2]...
    traj = np.empty((V.shape[-1], len(dyn_sys.x)))
    traj[0] = dyn_sys.x

    # flip the value with respect to the index
    V = np.flip(V, grid.dims)

    opt_u = []
    opt_d = []
    t = []
    t_earliest = -1

    t_earlist_log = []
    v_log = []
    n2p_log = []

    valueAtX = grid.get_value(V[..., 0], dyn_sys.x)
    if valueAtX > 0:
        V = V - valueAtX
    
    # the traj should start asap since we trust the FRAT timebound
    negToPos, posToNeg = find_sign_change(grid, V, dyn_sys.x)
    t_earliest = negToPos[0]

    for iter in range(0,len(tau)):
        if iter < t_earliest:
            traj[iter] = np.array(dyn_sys.x)
            t.append(tau[iter])
            t_earlist_log.append(t_earliest)
            v_log.append(grid.get_value(V[..., iter], dyn_sys.x))
            n2p_log.append(negToPos)
            continue
        
        # always track the edge closest(in time) to the current state
        negToPos, posToNeg = find_sign_change(grid, V, dyn_sys.x)
        n2p_log.append(negToPos)
        if negToPos.size != 0:
            valueAtX = grid.get_value(V[..., iter], dyn_sys.x)
            if valueAtX <= 0:
                for value in negToPos:
                    if value >= iter:
                        t_earliest = value
                        break
            else:
                for value in reversed(negToPos):
                    if value <= iter:
                        t_earliest = value
                        break

        t.append(tau[iter])

        brs_at_t = V[..., t_earliest]
        t_next = t_earliest + 1

        if t_next > V.shape[-1] - 1:
            t_next = V.shape[-1] - 1

        gradient = spa_deriv(grid.get_index(dyn_sys.x), brs_at_t, grid, periodic_dims=[2])

        for _ in range(subsamples):
            u = dyn_sys.opt_ctrl_non_hcl(_, dyn_sys.x, gradient)
            d = dyn_sys.opt_dstb_non_hcl(_, dyn_sys.x, gradient)
            dNone = [0,0,0]
            bestU = u
            # if minVal > 0:
            #     for tempU in uBranches:
            #         tempVal = grid.get_value(V[..., iter], next_state(dyn_sys, tempU, d, dt))
            #         if tempVal < minVal:
            #             minVal = tempVal
            #             bestU = tempU
            nextState = next_state(dyn_sys, bestU, dNone, dt)
            update_state(dyn_sys, nextState)
            opt_u.append(u)
            opt_d.append(d)

        t_earlist_log.append(t_earliest)
        v_log.append(grid.get_value(V[..., iter], dyn_sys.x))
        # the agent has entered the target
        if t_earliest == V.shape[-1]:
            # if arriveAfter is not None:
            #     if iter > arriveAfter:
            #         traj[iter:] = np.array(dyn_sys.x)
            #         break
            # else:
            traj[iter:] = np.array(dyn_sys.x)
            break

        if iter != V.shape[-1]:
            traj[iter] = np.array(dyn_sys.x)

    return traj, opt_u, opt_d, t

