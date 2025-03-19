import numpy as np
import math
from typing import List


class Grid:
    def __init__(
        self,
        minBounds: List,
        maxBounds: List,
        dims: int,
        pts_each_dim: List,
        periodicDims: List = [],
    ):
        """
        Args:
            minBounds (list): The lower bounds of each dimension in the grid
            maxBounds (list): The upper bounds of each dimension in the grid
            dims (int): The dimension of grid
            pts_each_dim (list): The number of points for each dimension in the grid
            periodicDim (list, optional): A list of periodic dimentions (0-indexed). Defaults to [].
        """
        assert len(minBounds) == len(maxBounds) == len(pts_each_dim) == dims

        self.max = np.array(maxBounds)
        self.min = np.array(minBounds)
        self.dims = dims
        self.pts_each_dim = np.array(pts_each_dim)
        self.pDim = periodicDims

        # Exclude the upper bounds for periodic dimensions is not included
        # e.g. [-pi, pi)
        for dim in self.pDim:
            self.max[dim] = self.min[dim] + (self.max[dim] - self.min[dim]) * (
                1 - 1 / self.pts_each_dim[dim]
            )
        self.dx = (self.max - self.min) / (self.pts_each_dim - 1.0)

        """
        Below is re-shaping the self.vs so that we can make use of broadcasting
        self.vs[i] is reshape into (1,1, ... , pts_each_dim[i], ..., 1) such that pts_each_dim[i] is used in ith position
        """
        self.vs = []
        """
        self.grid_points is same as self.vs; however, it is not reshaped. 
        self.grid_points[i] is a numpy array with length pts_each_dim[i] 
        """
        self.grid_points = []
        for i in range(dims):
            tmp = np.linspace(self.min[i], self.max[i], num=self.pts_each_dim[i])
            broadcast_map = np.ones(self.dims, dtype=int)
            broadcast_map[i] = self.pts_each_dim[i]
            self.grid_points.append(tmp)

            # in order to add our range of points to our grid
            # we need to modify the shape of tmp in order to match
            # the size of the grid for one of the axis
            tmp = np.reshape(tmp, tuple(broadcast_map))
            self.vs.append(tmp)

    def __str__(self):
        return (
            f"Grid:\n"
            + f"  max: {self.max}\n"
            + f"  min: {self.min}\n"
            + f"  pts_each_dim: {self.pts_each_dim}\n"
            + f"  pDim: {self.pDim}\n"
            + f"  dx: {self.dx}\n"
        )

    def get_index(self, state):
        """Returns a tuple of the closest index of each state in the grid

        Args:
            state (tuple): state of dynamic object
        """
        index = []

        for i, s in enumerate(state):
            idx = np.searchsorted(self.grid_points[i], s)
            if idx > 0 and (
                idx == len(self.grid_points[i])
                or math.fabs(s - self.grid_points[i][idx - 1])
                < math.fabs(s - self.grid_points[i][idx])
            ):
                index.append(idx - 1)
            else:
                index.append(idx)

        return tuple(index)

    def get_value(self, V, state):
        """Obtain the approximate value of a state

        Assumes that the state is within the bounds of the grid

        Args:
            V (np.array): value function of solved HJ PDE
            state (tuple): state of dynamic object

        Returns:
            [float]: V(state)
        """
        index = self.get_index(state)
        return V[index]
