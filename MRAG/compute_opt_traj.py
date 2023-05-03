from odp.Grid.GridProcessing import Grid
# from first_order_generic import spa_deriv
from update_state import next_state, update_state
import numpy as np
from find_tEarlist import find_sign_change

def compute_opt_traj1v0(grid, V, tau, dynamics, x1_1v0, x2_1v0, subsamples=1):
    """
    Computes the optimal trajectory, controls and disturbance to a minimal BRT/BRS based on the dynamics

    Args:
        the flip
        grid (class): instance
        V: [x1_grid, x2_grid, len(tau)]
        tau: 
        dynamics:
        subsamples: Number of times opt_u and opt_d are calculated within dt
        x1_1v0: spatial derivative array

    Returns:
        traj: State of dyn_sys from time T[-1] to T[0]
        opt_u: Optimal control at each time
        opt_d: Optimal disturbance at each time

    """
    assert V.shape[-1] == len(tau)

    dt = (tau[1] - tau[0]) / subsamples  # the subsample is recommended to be 1

    # first entry is dyn_sys at time tau[-1]
    # second entry is dyn_sys at time tau[-2]...
    traj = np.empty((V.shape[-1], len(dynamics.x)))  # traj.shape = [len(tau), dim]
    traj[0] = dynamics.x

    opt_u = []
    opt_d = []
    t = []
    t_earliest = -1

    t_earlist_log = []
    v_log = []
    n2p_log = []

    valueAtX = grid.get_value(V[..., 0], dynamics.x) # here check whether the initial position is in the RAS V[..., 0]
    if valueAtX > 0:  
        V = V - valueAtX
    
    # the traj should start asap since we trust the FRAT timebound
    negToPos, posToNeg = find_sign_change(grid, V, dynamics.x) 
    print("negToPos is {}".format(negToPos))
    t_earliest = negToPos  # to unify variable names

    for iter in range(0,len(tau)):
        if iter < t_earliest:
            traj[iter] = np.array(dynamics.x)  # before reaches the edge, the agent keeps still
            t.append(tau[iter])
            t_earlist_log.append(t_earliest)
            v_log.append(grid.get_value(V[..., iter], dynamics.x))
            n2p_log.append(negToPos)
            continue  # control=0
        
        # always track the edge closest(in time) to the current state
        negToPos, posToNeg = find_sign_change(grid, V, dynamics.x)
        n2p_log.append(negToPos)
        if negToPos.size != 0:
            valueAtX = grid.get_value(V[..., iter], dynamics.x)
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

        # gradient = spa_deriv(grid.get_index(dynamics.x), brs_at_t, grid, periodic_dims=[2])

        idx1, idx2 = grid.get_index(dynamics.x)
        print("idx1 {}, idx2 {}, t_earliest {}".format(idx1, idx2, t_earliest))
        # print(x1_1v0)
        spat_deriv_vector = (x1_1v0[idx1, idx2, t_earliest], x2_1v0[idx1, idx2, t_earliest])

        print("spat deriv is {}".format(x1_1v0[idx1, idx2, t_earliest]))
        for _ in range(subsamples):
            u1, u2 = dynamics.optCtrl_inPython(spat_deriv_vector)
            d1, d2 = dynamics.optDstb_inPython(spat_deriv_vector)
            dNone = [d1, d2]
            bestU = [u1, u2]
            # if minVal > 0:
            #     for tempU in uBranches:
            #         tempVal = grid.get_value(V[..., iter], next_state(dyn_sys, tempU, d, dt))
            #         if tempVal < minVal:
            #             minVal = tempVal
            #             bestU = tempU
            nextState = next_state(dynamics, bestU, dNone, dt)
            print("next state is {}".format(nextState))
            update_state(dynamics, nextState)
            opt_u.append((u1, u2))
            opt_d.append((d1, d2))

        t_earlist_log.append(t_earliest)
        v_log.append(grid.get_value(V[..., iter], dynamics.x))
        # the agent has entered the target
        if t_earliest == V.shape[-1]:
            # if arriveAfter is not None:
            #     if iter > arriveAfter:
            #         traj[iter:] = np.array(dyn_sys.x)
            #         break
            # else:
            traj[iter:] = np.array(dynamics.x)
            break

        if iter != V.shape[-1]:
            print(dynamics.x)
            traj[iter] = np.array(dynamics.x)
    return traj, opt_u, opt_d, t

