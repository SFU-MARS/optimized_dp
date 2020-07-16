import numpy as np
from Grid.GridProcessing import grid
from Shapes.ShapesFunctions import *

# Specify the  file that includes dynamic systems
from Plots.plotting_utilities import *

import math

def main():
    g = grid(np.array([-5.0, -5.0, -1.0, -math.pi]), np.array([5.0, 5.0, 1.0, math.pi]), 4, np.array([40, 40, 50, 50]), [3])

    V_1 = np.load("v1_brt.npy")

    plot_isosurface(g, V_1, [0, 1, 3])

if __name__ == '__main__':
    main()