import numpy as np


def compute_disturbance(dynamics, grids, values, state):
    """Computes the optimal disturbance given the current state.

    Args:
        dynamics: (instance): The instance of the given dynamics.
        grids (instance): The instance of the class Grid.
        values (ndarray): The final time slice value function, in the shape of [size, size, ..., 1]
        state (tuple): The current state.
    
    Returns:
        The current diturbance inputs.

    """
    spat_deriv_vector = spa_deriv(grids.get_index(state), values, grids)
    return dynamics.optDstb_inPython(spat_deriv_vector)


def compute_control(dynamics, grids, values_all, tau, state):
    """Computes the optimal control given the current state.

    Args:
        dynamics: (instance): The instance of the given dynamics.
        grids (instance): The instance of the class Grid.
        values (ndarray): The value function with all time slices, in the shape of [size, size, ..., len(tau)].
        tau (ndarray): All time indices.
        state (tuple): The current state.
    
    Returns:
        The current control inputs.
    """
    neg2pos, pos2neg = find_sign_change(grids, values_all, state, tau)
    num_state = len(state)  # suppose the dimension of the state equals to the dimension of the control inputs
    if len(neg2pos):
        assert values_all.shape[-1] == len(tau)  
        current_value = grids.get_value(values_all[..., 0], list(state))
        if current_value > 0:
            value1v0 = value1v0 - current_value
        values = values_all[..., neg2pos]
        spat_deriv_vector = spa_deriv(grids.get_index(state), values, grids)
        return dynamics.optCtrl_inPython(spat_deriv_vector)
    else:
        return tuple([0.0] * num_state)

def find_sign_change(grids, values, state, tau):
    """Finds the positions of the positive and negative transformation of the values.

    Args:
    grids (instance): The instance of the class Grid.
    values (ndarray): The value function with all time slices, shapes like [size, size, ..., len(tau)].
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