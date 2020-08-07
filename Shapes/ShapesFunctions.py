import numpy as np

# This function creates a cyclinderical shape
def CyclinderShape3D(grid, ignore_dim, center, radius):
    data = np.zeros(grid.pts_each_dim)
    for i in range(0, 3):
        if i != ignore_dim-1:
            # This works because of broadcasting
            data = data + np.power(grid.vs[i] - center[i],  2)
    data = np.sqrt(data) - radius
    return data


def Cylinder6D(grid, ignore_dims):
    data = np.zeros(grid.pts_each_dim)
    for i in range(0, 6):
        if i + 1 not in ignore_dims:
            # This works because of broadcasting
            data = data + np.power(grid.vs[i], 2)
    data = np.sqrt(data)
    return data


def Cylinder4D(grid, ignore_dims):
    data = np.zeros(grid.pts_each_dim)
    for i in range(0, 4):
        if i + 1 not in ignore_dims:
            # This works because of broadcasting
            data = data + np.power(grid.vs[i], 2)
    data = np.sqrt(data)
    return data


def CylinderShape(grid, ignore_dims, center, radius):
    """Creates an axis align cylinder implicit surface function

    Args:
        grid (Grid): Grid object
        ignore_dims (List) : List  specifing axis where cylindar is aligned (1-indexed)
        center (List) :  List specifing the center of cylinder 
        radius (float): Radius of cylinder

    Returns:
        np.ndarray: implicit surface function of the cylinder
    """
    data = np.zeros(grid.pts_each_dim)
    for i in range(grid.dims):
        if i + 1 not in ignore_dims:
            # This works because of broadcasting
            data = data + np.power(grid.vs[i] - center[i], 2)
    data = np.sqrt(data) - radius
    return data

# Range is a list of list of ranges
# def Rectangle4D(grid, range):
#     data = np.zeros(grid.pts_each_dim)
#
#     for i0 in (0, grid.pts_each_dim[0]):
#         for i1 in (0, grid.pts_each_dim[1]):
#             for i2 in (0, grid.pts_each_dim[2]):
#                 for i3 in (0, grid.pts_each_dim[3]):
#                     x0 = grid.xs[i0]
#                     x1 = grid.xs[i1]
#                     x2 = grid.xs[i2]
#                     x3 = grid.xs[i3]
#                     range_list = [-x0 + range[0][0], x0 - range[0][1],
#                                   -x1 + range[1][0], x1 - range[1][1],
#                                   -x2 + range[2][0], x2 - range[2][1],
#                                   -x3 + range[3][0], x3 - range[3][1]]
#                     data[i0, i1, i2, i3] = min(range_list)
#     return data


def Rectangle6D(grid):
    #data = np.zeros(grid.pts_each_dim)
    #x1 = np.reshape(grid.vs[0], (25, 25, 25, 25, 25, 25))
    data = np.maximum(grid.vs[0] - 0.05, -grid.vs[0] - 0.05)
    data = np.maximum(data,  grid.vs[1] - 0.08)
    data = np.maximum(data, -grid.vs[1] - 0.08)
    data = np.maximum(data,  grid.vs[2] - 1.04)
    data = np.maximum(data, -grid.vs[2] + 0.94)
    data = np.maximum(data,  grid.vs[3] - 0.15)
    data = np.maximum(data, -grid.vs[3] - 0.15)
    data = np.maximum(data,  grid.vs[4] - 0.15)
    data = np.maximum(data, -grid.vs[4] - 0.15)
    data = np.maximum(data,  grid.vs[5] - 0.6)
    data = np.maximum(data, -grid.vs[5] - 0.6)
    return data


def ShapeRectangle(grid, target_min, target_max):
    data = np.maximum(grid.vs[0] - target_max[0], -grid.vs[0] + target_min[0])

    for i in range(1, grid.dims):
        data = np.maximum(data,  grid.vs[i] - target_max[i])
        data = np.maximum(data, -grid.vs[1] + target_min[i])

    return data


def Rect_Around_Point(grid, target_point):
    return ShapeRectangle(grid, target_point - 1.5*grid.dx, target_point + 1.5*grid.dx)


def Lower_Half_Space(grid, dim, value):
    """Creates an axis aligned lower half space 

    Args:
        grid (Grid): Grid object
        dim (int): Dimention of the half space (1-indexed)
        value (float): Used in the implicit surface function for V < value

    Returns:
        np.ndarray: implicit surface function of the lower half space
                    of size grid.pts_each_dim
    """
    data = np.zeros(grid.pts_each_dim)
    for i in range(1, grid.dims + 1):
        if i == dim:
            data += grid.vs[i] - value
    return data


def Upper_Half_Plane(grid, dim, value):
    """Creates an axis aligned lower half space 

    Args:
        grid (Grid): Grid object
        dim (int): Dimention of the half space (1-indexed)
        value (float): Used in the implicit surface function for V > value

    Returns:
        np.ndarray: implicit surface function of the lower half space
                    of size grid.pts_each_dim
    """
    data = np.zeros(grid.pts_each_dim)
    for i in range(1, grid.dims + 1):
        if i == dim:
            data += -grid.vs[i] + value
    return data
