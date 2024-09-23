import numpy as np


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


def check_target(grids, target, current_state):
    current_slice = grids.get_index(current_state)
    current_value = target[current_slice]
    # print(f"The current value in the check_target function is {current_value}")
    if current_value > 0:
        return False
    else:
        return True


def compute_opt_traj_TTR(dynamics, grids, TTR_value, start_state, target, ctrl_freq, periodic_dims=[]):
    """Computes the trajectory with respect to the optimal control based on the TTR value function and the dynamics.

    Args:
        dynamics (instance): The instance of the given dynamics.
        grids (instance): The instance of the class Grid.
        values_all (ndarray): The value function with all time slices, in the shape of [grid, grid, ..., len(tau)].
        tau (ndarray): All time indices.
        state (tuple): The current state.
        ctrl_freq (int): The control frequency in Hz.
    
    Returns:
        traj (list): The trajectory of the agent.
        opt_u (list): The optimal control inputs of the agent.
        opt_d (list): The optimal disturbance.
        t (list): The time iterations.
    """
    # Get the time-to-reach value
    current_state = start_state
    current_value = grids.get_value(TTR_value, current_state)
    num_control = int(current_value*ctrl_freq)
    # Initializations
    traj = np.empty((num_control, len(start_state)))  # current_value*ctrl_freq is the number of control to be applied
    opt_u = []

    for t in range(num_control):
        # Log the state to the trajectory
        traj[t] = current_state
        # Check the state
        if check_target(grids, target, current_state):
            traj[t:] = current_state
            break
        # Compute the optimal control
        spat_deriv_vector = spa_deriv(grids.get_index(current_state), TTR_value[..., np.newaxis], grids, periodic_dims=periodic_dims)
        current_u = dynamics.optCtrl_inPython(spat_deriv_vector)
        opt_u.append(current_u)
        # Apply the control and update the current_state
        next_state = dynamics.forward(ctrl_freq, current_state, current_u)
        current_state = next_state
         
    print("The planning is over")
    return traj