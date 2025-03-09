import numpy as np
from odp.grid import Grid
from typing import List

"""
Functions for easily defining implicit surface functions for common shapes, 
taking the convention that the zero sublevel set of the implicit surface
function represents the set.

These functions also serve as examples of how to implement implicit surface
functions in general.
"""

def CylinderShape(
        grid: Grid, 
        ignore_dims: List, 
        center: List, 
        radius: float,
        quadratic: bool = True
        ) -> np.ndarray:
    """
    Creates an axis-aligned cylinder implicit surface function

    Args:
        grid (Grid): Grid object
        ignore_dims (List) : List specifing axis where cylinder is aligned (0-indexed)
        center (List) :  List specifying the center of cylinder
        radius (float): Radius of cylinder
        quadratic: (bool): Uses a quadratic function if True (linear if False)
                           Setting this to True can reduce numerical errors 
                           from dissipation

    Returns:
        np.ndarray: implicit surface function of the cylinder
    """
    data = np.zeros(grid.pts_each_dim)
    for i in range(grid.dims):
        if i not in ignore_dims:
            # This works because of broadcasting
            data = data + np.power(grid.vs[i] - center[i], 2)

    if quadratic:
        return data - radius*radius
    
    else:
        return np.sqrt(data) - radius
    

def ShapeRectangle(
        grid: Grid, 
        target_min: np.ndarray, 
        target_max: np.ndarray
        ) -> np.ndarray:
    data = Intersection(Lower_Half_Space(grid, 0, target_max[0]),
                        Upper_Half_Space(grid, 0, target_min[0])
                        )
    for i in range(1, grid.dims):
        data = Intersection(data, Lower_Half_Space(grid, i, target_max[i]))
        data = Intersection(data, Upper_Half_Space(grid, i, target_min[i]))

    return data


def Rect_Around_Point(grid: Grid, target_point: np.ndarray) -> np.ndarray:
    """
    Creates a small rectangle of width 3 grid points in each dimension around 
    target_point
    """
    return ShapeRectangle(grid, target_point - 1.5 * grid.dx, target_point + 1.5 * grid.dx)


def Lower_Half_Space(grid: Grid, dim: int, value: float) -> np.ndarray:
    """Creates an axis aligned lower half space 

    Args:
        grid (Grid): Grid object
        dim (int): Dimension of the half space (0-indexed)
        value (float): Used in the implicit surface function for V < value

    Returns:
        np.ndarray: implicit surface function of the lower half space
                    of size grid.pts_each_dim
    """
    data = np.zeros(grid.pts_each_dim)
    for i in range(grid.dims):
        if i == dim:
            data += grid.vs[i] - value
    return data


def Upper_Half_Space(grid: Grid, dim: int, value: float) -> np.ndarray:
    """Creates an axis aligned upper half space 

    Args:
        grid (Grid): Grid object
        dim (int): Dimension of the half space (0-indexed)
        value (float): Used in the implicit surface function for V > value

    Returns:
        np.ndarray: implicit surface function of the lower half space
                    of size grid.pts_each_dim
    """
    data = np.zeros(grid.pts_each_dim)
    for i in range(grid.dims):
        if i == dim:
            data += -grid.vs[i] + value
    return data


def Union(shape1: np.ndarray, shape2: np.ndarray) -> np.ndarray:
    """ Calculates the union of two shapes

    Args:
        shape1 (np.ndarray): implicit surface representation of a shape
        shape2 (np.ndarray): implicit surface representation of a shape

    Returns:
        np.ndarray: the element-wise minimum of two shapes
    """
    return np.minimum(shape1, shape2)


def Intersection(shape1: np.ndarray, shape2: np.ndarray) -> np.ndarray:
    """ Calculates the intersection of two shapes

    Args:
        shape1 (np.ndarray): implicit surface representation of a shape
        shape2 (np.ndarray): implicit surface representation of a shape

    Returns:
        np.ndarray: the element-wise minimum of two shapes
    """
    return np.maximum(shape1, shape2)


def ShapeEllipsoid(
        grid: Grid, 
        center: np.ndarray, 
        semiAxLen: List,
        quadratic: bool = True
        ) -> np.ndarray:
    """
    Creates an axis-aligned ellipsoid implicit surface function

    Args:
        grid (Grid): Grid object
        center (List) :  List specifying the center of the ellipsoid
        semiAxLen (List): List specifying the semi-axis lengths
        quadratic: (bool): Uses a quadratic function if True (linear if False)
                           Setting this to True can reduce numerical errors 
                           from dissipation

    Returns:
        np.ndarray: implicit surface function of the ellipsoid    
    """    
    data = np.zeros(grid.pts_each_dim)  

    for i in range(grid.dims):
        data = data + np.power(grid.vs[i] - center[i],2)/np.power(semiAxLen[i],2)

    if quadratic:
        return data - 1
    
    else:
        return np.sqrt(data) - 1

     