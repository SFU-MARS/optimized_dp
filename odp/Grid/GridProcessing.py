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

    def get_indices(self, states: np.ndarray) -> np.ndarray:
        """Returns a tuple of the closest indices of each state in the grid

        Args:
            states (np.ndarray): states of dynamical system, shape (N, self.dims)

        Returns:
            np.ndarray: indices of each state, shape (N, self.dims)
        """
        n = states.ndim
        if n == 1:
            states = np.array([states])

        assert states.shape[1] == self.dims
        
        indices = np.zeros(states.shape)

        for i in range(self.dims):
            states_i = states[:, i]  # Shape (N,)
            indices_i = np.searchsorted(self.grid_points[i], states_i)  # Shape (N,)
            indices_i_m1 = (indices_i - 1).astype(int)

            # Decrement indices that are at the upper edge, or that are closer to
            # the previous grid point than the next
            decrement_these = indices_i > 0 & (
                (indices_i == len(self.grid_points[i]))
                | (
                    (states_i - self.grid_points[i][indices_i_m1])
                    < (self.grid_points[i][indices_i] - states_i)
                )
            )
            indices_i[decrement_these] -= 1
            indices[:, i] = indices_i

        if n == 1:
            indices = indices[0]

        return tuple(indices.astype(int).T)

    def get_values(self, V: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Interpolates value function V on states (nearest neighbour). Assumes that
        all states are within the bounds of the grid

        Returns:
            [float]: V(state)

        Args:
            V (np.ndarray): value function of solved HJ PDE, shape self.pts_each_dim
            states (np.ndarray): states, shape (N, self.dims) or (self.dims,)

        Returns: V evaluated at all the states
            np.ndarray of shape (N,) if states is shape (N, self.dims)
            float if states is shape (self.dims,)
        """
        indices = self.get_indices(states)
        print(indices)
        return V[indices]
