import numpy as np
from GridProcessing import *
# Add initial value function to file ShapesFunction and include ShapesFunction in here  
from ShapesFunctions import *
# Specify the  file that includes dynamic systems
from Humannoid6D_sys1 import *
import scipy.io as sio

import math

""" USER INTERFACES
- Define grid

- Generate initial values for grid using shape functions

- Time length for computations

- Run
"""

# Grid field in this order: x, x_dot, z, z_dot, theta, theta_dot

g = grid(np.array([-0.5, -1.0, 0.5, -2.0, -math.pi/2, -8.0]), np.array([0.5, 1.0, 1.5, 2.0, math.pi/2, 8.0]), 6, np.array([27, 26, 27, 26, 27, 26])) # Leave out periodic field
# Define my object
my_humanoid = Humanoid_6D()

# Use the grid to initialize initial value function
my_shape = Rectangle6D(g)

# Look-back lenght and time step
lookback_length = 2.03
t_step = 0.05

tau = np.arange(start = 0, stop = lookback_length + t_step, step = t_step)
print("I'm here \n")

# Use the following variable to specify the characteristics of computation
compMethod = "minVWithVInit"
my_object  = my_humanoid



