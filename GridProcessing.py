import numpy as np

class grid:
  def __init__(self, min, max, dims ,pts_each_dim, pDim):
        self.max = max
        self.min = min
        self.dims = dims
        self.pts_each_dim = pts_each_dim
        self.pDim = pDim

        # Make some modifications to the initialized
        self.max[pDim] = self.min[pDim] + (self.max[pDim] - self.min[pDim])  * (1 - 1/self.pts_each_dim[pDim])
        self.dx = (self.max - self.min) / (self.pts_each_dim - 1.0)

        """
        Below is re-shaping the self.vs so that we can make use of broadcasting
        self.vs[i] is reshape into (1,1, ... , pts_each_dim[i], ..., 1) such that pts_each_dim[i] is used in ith position
        """
        self.vs = []
        for i in range(0,dims):
            tmp = np.linspace(self.min[i],self.max[i], num=self.pts_each_dim[i])
            broadcast_map = np.ones(self.dims, dtype=int)
            broadcast_map[i] = self.pts_each_dim[i]
            tmp = np.reshape(tmp, tuple(broadcast_map))
            self.vs.append(tmp)

        # Turn pts_each_dim to complex numbers to input into meshgrid
        complex_x = complex(0, pts_each_dim[0])
        complex_y = complex(0, pts_each_dim[1])
        complex_z = complex(0, pts_each_dim[2])

        # Grid 's meshgrid
        self.mg_X, self.mg_Y, self.mg_T = np.mgrid[self.min[0]:self.max[0]: complex_x, self.min[1]:self.max[1]: complex_y, self.min[2]:self.max[2]: complex_z]
