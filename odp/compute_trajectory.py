import numpy as np


def find_sign_change(grids, values, state, tau):
    """Finds the positions of the positive and negative transformation of the values.

    Args:
    grids (instance): The instance of the class Grid.
    values (ndarray): The value function with all time slices, shapes like [grid, grid, ..., len(tau)].
    state (tuple): The current state.
    tau (ndarray): All time indices.

    Returns:
        The two positions (neg2pos, pos2neg) where the transformation happens.
    """
    current_slices = grids.get_index(state)
    indices = tuple([idx for idx in current_slices] + [slice(0, len(tau))])
    current_value = values[indices]
    neg_values = (current_value<=0).astype(int) 
    checklist = neg_values - np.append(neg_values[1:], neg_values[-1])
    return np.where(checklist==1)[0], np.where(checklist==-1)[0]


def spa_deriv(indices, values, grids, periodic_dims=[]):
    """Calculates the spatial derivatives of the values at an index for each dimension

    Args:
        indices (tuple): The indices of the state in the grid dimension.
        values (ndarray): The value function shapes like [..., neg2pos] where neg2pos is a list [scalar] or [].
        grids (class): The instance of the corresponding Grid.
        periodic_dims (list): The corrsponding periodical dimensions [].

    Returns:
        List of left and right spatial derivatives for each dimension.
    """
    spa_derivatives = []
    for dim, idx in enumerate(indices):
        if dim == 0:
            left_index = []
        else:
            left_index = list(indices[:dim])

        if dim == len(indices) - 1:
            right_index = []
        else:
            right_index = list(indices[dim + 1:])

        next_index = tuple(
            left_index + [indices[dim] + 1] + right_index
        )
        prev_index = tuple(
            left_index + [indices[dim] - 1] + right_index
        )

        if idx == 0:
            if dim in periodic_dims:
                left_periodic_boundary_index = tuple(
                    left_index + [values.shape[dim] - 1] + right_index
                )
                left_boundary = values[left_periodic_boundary_index]
            else:
                left_boundary = values[indices] + np.abs(values[next_index] - values[indices]) * np.sign(values[indices])
            left_deriv = (values[indices] - left_boundary) / grids.dx[dim]
            right_deriv = (values[next_index] - values[indices]) / grids.dx[dim]
        elif idx == values.shape[dim] - 1:
            if dim in periodic_dims:
                right_periodic_boundary_index = tuple(
                    left_index + [0] + right_index
                )
                right_boundary = values[right_periodic_boundary_index]
            else:
                right_boundary = values[indices] + np.abs(values[indices] - values[prev_index]) * np.sign([values[indices]])
            left_deriv = (values[indices] - values[prev_index]) / grids.dx[dim]
            right_deriv = (right_boundary - values[indices]) / grids.dx[dim]
        else:
            left_deriv = (values[indices] - values[prev_index]) / grids.dx[dim]
            right_deriv = (values[next_index] - values[indices]) / grids.dx[dim]

        spa_derivatives.append(((left_deriv + right_deriv) / 2)[0])
    return spa_derivatives


def next_position(dynamics, current_state, u, d, tstep):
    """Updates the next position of the agent.

    Arg:
        dynamics (instance): The instance of the given dynamics.
        current_state (tuple): The current state of the agent.
        u (tuple): The current control inputs u.
        d (tuple): The current disturbance inputs d.
        tstep (int): The time interval.

    Returns:
        The new position.
    """
    next_state = current_state + tstep * dynamics.dynamics_python(current_state, u, d)
    return next_state


def compute_opt_traj(dynamics, grids, values_all, tau, state):
    """Computes the trajectory, optimal controls and disturbances to the BRT based on the dynamics in the length of time slices.

    Args:
        dynamics (instance): The instance of the given dynamics.
        grids (instance): The instance of the class Grid.
        values_all (ndarray): The value function with all time slices, in the shape of [grid, grid, ..., len(tau)].
        tau (ndarray): All time indices.
        state (tuple): The current state.
    
    Returns:
        traj (list): The trajectory of the agent.
        opt_u (list): The optimal control inputs of the agent.
        opt_d (list): The optimal disturbance.
        t (list): The time iterations.
    """
    assert values_all.shape[-1] == len(tau)
    print(f"The number of steps is {len(tau)}.")
    dt = tau[1] - tau[0]

    current_state = state
    traj = np.empty((values_all.shape[-1], len(state)))  # traj.shape = [len(tau), dim]
    traj[0] = current_state  # dynamics.x
    t_earliest = -1

    opt_u = []
    opt_d = []
    t = []
    t_earlist_log = []
    v_log = []
    n2p_log = []

    current_value = grids.get_value(values_all[..., 0], current_state) # here check whether the initial position is in the RAS V[..., 0]
    if current_value > 0:  
        values_all = values_all - current_value
    
    # calculate the time slice where the value changes from positive to negative
    negToPos, posToNeg = find_sign_change(grids, values_all, current_state, tau) 
    t_earliest = negToPos 

    for iter in range(0, len(tau)):
        if iter < t_earliest:
            # stand still
            traj[iter] = np.array(current_state)  # before reaches the edge, the agent keeps still
            t.append(tau[iter])
            t_earlist_log.append(t_earliest)
            v_log.append(grids.get_value(values_all[..., iter], current_state))
            n2p_log.append(negToPos)
            opt_u.append(0)
            continue  # control=0
        
        # we should apply the control inputs to the agent
        negToPos, posToNeg = find_sign_change(grids, values_all, current_state, tau)
        n2p_log.append(negToPos)

        if negToPos.size != 0:
            # the agent has not arrived at the target
            current_value = grids.get_value(values_all[..., iter], current_state)
            if current_value <= 0:  # the agent is within the BRT/BRS
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
            t_next = t_earliest + 1

            if t_next > values_all.shape[-1] - 1:
                t_next = values_all.shape[-1] - 1

            # calculate the control inputs
            values = values_all[..., [t_earliest]]
            spat_deriv_vector = spa_deriv(grids.get_index(current_state), values, grids)
            current_u = dynamics.optCtrl_inPython(spat_deriv_vector)
            current_d = dynamics.optDstb_inPython(spat_deriv_vector)
            opt_u.append(current_u)
            opt_d.append(current_d)

            # apply the control and update the current_state
            next_state = next_position(dynamics, current_state, current_u, current_d, dt)
            current_state = next_state
            t_earlist_log.append(t_earliest)
            v_log.append(grids.get_value(values_all[..., iter], current_state))
            if iter != values_all.shape[-1]:
                traj[iter] = np.array(current_state)
        else:  
            # the agent has arrived at the target
            traj[iter:] = np.array(current_state)
            break

    return traj, opt_u, opt_d, t