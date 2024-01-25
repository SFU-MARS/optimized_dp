import numpy as np
from .grid import Grid

def cylinder(grid: Grid, center, axes, radius):
    """Creates an axis align cylinder implicit surface function

    Args:
        grid (Grid): Grid object
        ignore_dims (List): List specifing axis where cylinder is aligned (0-indexed)
        center (List): List specifying the center of cylinder
        radius (float): Radius of cylinder

    Returns:
        np.ndarray: implicit surface function of the cylinder
    """
    data = np.zeros(grid.shape)
    for i in range(grid.ndims):
        if i not in axes:
            # This works because of broadcasting
            data = data + np.power(grid.vs[i] - center[i], 2)
    data = np.sqrt(data) - radius
    return data


def rectangle(grid: Grid, target_min, target_max):
    data = np.maximum(grid.vs[0] - target_max[0], -grid.vs[0] + target_min[0])

    for i in range(grid.dims):
        data = np.maximum(data,  grid.vs[i] - target_max[i])
        data = np.maximum(data, -grid.vs[i] + target_min[i])

    return data


def point(grid: Grid, target_point):
    return rectangle(grid, target_point - 1.5 * grid.dx, target_point + 1.5 * grid.dx)


def lower_half_space(grid: Grid, axis, value):
    """Creates an axis aligned lower half space 

    Args:
        grid (Grid): Grid object
        axis (int): Dimension of the half space (0-indexed)
        value (float): Used in the implicit surface function for V < value

    Returns:
        np.ndarray: implicit surface function of the lower half space
                    of size grid.pts_each_dim
    """
    data = np.zeros(grid.shape)
    for i in range(grid.ndims):
        if i == axis:
            data += grid.vs[i] - value
    return data


def upper_half_space(grid: Grid, axis, value):
    """Creates an axis aligned upper half space 

    Args:
        grid (Grid): Grid object
        axis (int): Dimension of the half space (0-indexed)
        value (float): Used in the implicit surface function for V > value

    Returns:
        np.ndarray: implicit surface function of the lower half space
                    of size grid.pts_each_dim
    """
    data = np.zeros(grid.shape)
    for i in range(grid.ndims):
        if i == axis:
            data += -grid.vs[i] + value
    return data


def union(shape, *shapes):
    """ Calculates the union of two shapes

    Args:
        shape1 (np.ndarray): implicit surface representation of a shape
        shape2 (np.ndarray): implicit surface representation of a shape

    Returns:
        np.ndarray: the element-wise minimum of two shapes
    """
    result = shape 
    for shape in shapes: 
        result = np.minimum(result, shape)
    return result


def intersection(shape, *shapes):
    """ Calculates the intersection of two shapes

    Args:
        shape1 (np.ndarray): implicit surface representation of a shape
        shape2 (np.ndarray): implicit surface representation of a shape

    Returns:
        np.ndarray: the element-wise minimum of two shapes
    """
    result = shape
    for shape in shapes:
        result = np.maximum(result, shape)
    return result

