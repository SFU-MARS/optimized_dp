import numpy as np
from Grid.GridProcessing import grid
from Shapes.ShapesFunctions import *

# Specify the  file that includes dynamic systems
from dynamics.DubinsCar4D import *
import scipy.io as sio

import math

""" USER INTERFACES
- Define grid

- Generate initial values for grid using shape functions

- Time length for computations

- Run
"""

# Grid field in this order: min_range, max_range, number of dims, grid dimensions, list of periodic dim: starting at 0
g = grid(np.array([-5.0, -5.0, -0.5, -math.pi]), np.array([5.0, 5.0, 0.5, math.pi]), 4, np.array([40, 40, 50, 50]), [3])

# Define my object
my_car = DubinsCar4D()

# Use the grid to initialize initial value function
Initial_value_f = Cylinder4D(g, [3,4])

# Look-back lenght and time step
lookback_length = 2.03
t_step = 0.05

tau = np.arange(start = 0, stop = lookback_length + t_step, step = t_step)
print("I'm here \n")

# Use the following variable to specify the characteristics of computation
compMethod = "None"
my_object  = my_car
my_shape = Initial_value_f


