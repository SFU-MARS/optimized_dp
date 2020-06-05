import numpy as np

class grid:
  def __init__(self, min, max, dims ,pts_each_dim, pDim=[]):
        self.max = max
        self.min = min
        self.dims = len(pts_each_dim)
        self.pts_each_dim = pts_each_dim
        self.pDim = pDim

        # Make some modifications to the initialized
        for dim in pDim:
            self.max[dim] = self.min[dim] + (self.max[dim] - self.min[dim]) * (1 - 1/self.pts_each_dim[dim])
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
