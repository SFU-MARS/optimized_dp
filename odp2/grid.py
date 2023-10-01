import numpy as np

class Grid:
    """
    Fields:
        shape (tuple[int, ...]): Number of points in each dimension axis.
        pts_each_dim (None): Deprecated, use shape.
        ndims (int): Number of dimensions in grid.
        dims (None): Deprecated, use ndims.
        max_bounds (np.ndarray): Upper bounds of each dimension.
        max (None): Deprecated, use max_bounds.
        min_bounds (np.ndarray): Lower bounds of each dimension.
        min (None): Deprecated, use min_bounds.
        periodic_dims (Optional[list[bool]]): 
            Boolean switches for each axes if periodic or not. Defaults to all
            false, i.e. non-periodic.
        pDim (None): Deprecated, use periodic_dims.
        dx (np.ndarray): Discretization step for each dimension.
    """
    
    def __init__(self, shape, max_bounds, min_bounds, periodic_dims=None):
        """ 

        Args:
            shape (list): The number of points for each dimension in the grid.
            max_bounds (list): The upper bounds of each dimension in the grid.
            min_bounds (list): The lower bounds of each dimension in the grid.
            periodic_dims(list, optional): 
                Boolean switches for each axes if periodic or not. Defaults to
                all false, i.e. non-periodic.
        """
        
        self.shape = tuple(map(int, shape))
        self.ndims = len(shape)

        self.max_bounds = np.asarray(max_bounds)
        self.min_bounds = np.asarray(min_bounds)
        self.periodic_dims = [] if periodic_dims is None else periodic_dims

        # Exclude the upper bounds for periodic dimensions is not included 
        # e.g. [-pi, pi)
        for dim in self.periodic_dims:
            self.max_bounds[dim] = (self.min_bounds[dim]
                                    + ((self.max_bounds[dim] - self.min_bounds[dim])
                                       * (1 - 1/self.shape[dim])))
        self.dx = (self.max_bounds - self.min_bounds) / (np.array(self.shape) - 1.0)

        """
        Below is re-shaping the self.vs so that we can make use of broadcasting
        self.vs[i] is reshape into (1,1, ... , shape[i], ..., 1) such that shape[i] is used in ith position
        """
        self.vs = []
        """
        self.grid_points is same as self.vs; however, it is not reshaped. 
        self.grid_points[i] is a numpy array with length shape[i] 
        """
        self.grid_points = []
        for i in range(self.ndims):
            tmp = np.linspace(self.min_bounds[i], self.max_bounds[i],
                              num=self.shape[i])
            broadcast_map = np.ones(self.ndims, dtype=int)
            broadcast_map[i] = self.shape[i]
            self.grid_points.append(tmp)

            # in order to add our range of points to our grid
            # we need to modify the shape of tmp in order to match
            # the size of the grid for one of the axis
            tmp = np.reshape(tmp, tuple(broadcast_map))
            self.vs.append(tmp)

    def get_index(self, state):
        """ Returns a tuple of the closest index of each state in the grid

        Args:
            state (tuple): state of dynamic object
        """
        index = []

        for i, s in enumerate(state):
            idx = np.searchsorted(self.grid_points[i], s)
            if idx > 0 and (
                idx == len(self.grid_points[i])
                or np.fabs(s - self.grid_points[i][idx - 1])
                < np.fabs(s - self.grid_points[i][idx])
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
