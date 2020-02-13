import numpy as np

# This function creates a cyclinderical shape
def CyclinderShape(grid, ignore_dim, center, radius):
    data = np.zeros(grid.pts_each_dim)

    for i in range (0, 3):
        if i != ignore_dim-1:
            # This works because of broadcasting
            data = data + np.power(grid.vs[i] - center[i],  2)
    data = np.sqrt(data) - radius
    return data