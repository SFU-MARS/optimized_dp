def spa_deriv(index: Tuple, V, g):
    """
    Calculates the spatial derivatives of V at an index for each dimension

    Args:
        index:
        V:
        g:

    Returns:
        List of left and right spatial derivatives for each dimension

    """
    spa_derivatives = []
    for dim, idx in enumerate(index):
        if dim == 0:
            left_index = []
        else:
            left_index = list(index[:dim])

        if dim == len(index) - 1:
            right_index = []
        else:
            right_index = list(index[dim + 1:])

        next_index = tuple(
            left_index + [index[dim] + 1] + right_index
        )
        prev_index = tuple(
            left_index + [index[dim] - 1] + right_index
        )

        if idx == 0:
            left_boundary = V[index] + np.abs(V[next_index] - V[index]) * np.sign(V[index])
            left_deriv = (V[index] - left_boundary) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]
        elif idx == V.shape[dim] - 1:
            prev_index = tuple(
                left_index + [index[dim] - 1] + right_index
            )
            right_boundary = V[index] + np.abs(V[index] - V[prev_index]) * np.sign([V[index]])
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (right_boundary - V[index]) / g.dx[dim]
        else:
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]

        spa_derivatives.append((left_deriv + right_deriv) / 2)

    return spa_derivatives



