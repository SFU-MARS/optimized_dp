import numpy as np
from Grid.GridProcessing import grid
from Shapes.ShapesFunctions import *

# Specify the  file that includes dynamic systems
from dynamics.Humanoid12D_sys1 import *
import scipy.io as sio

import math

""" USER INTERFACES
- Define grid

- Generate initial values for grid using shape functions

- Time length for computations

- Run
"""

# Grid field in this order: min_range, max_range, number of dims, grid dimensions, list of periodic dim: starting at 0

g = grid(np.array([-0.5, -1, -0.5, -1, -0.5, -5]), \
         np.array([ 0.5,  1,  0.5,  1,  0.5,  5]), \
         6, \
         np.array([21, 21, 21, 21, 21, 21]), \
        )

# Define my object
dyn_sys = Humanoid12D_sys2()

# Use the grid to initialize initial value function
target_min = np.array([-0.05, -1.5*g.dx[1], -0.05, -1.5*g.dx[3], 1-1.5*g.dx[4], -1.5*g.dx[5]])
target_max = np.array([ 0.05,  1.5*g.dx[1],  0.05,  1.5*g.dx[3], 1+1.5*g.dx[4],  1.5*g.dx[5]])
Initial_value_f = ShapeRectangle(g, target_min, target_max)

# Look-back lenght and time step
lookback_length = 2.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start = 0, stop = lookback_length + small_number, step = t_step)
print("Welcome to optimized_dp \n")

# Use the following variable to specify the characteristics of computation
compMethod = "minVWithVInit"
my_object  = dyn_sys
my_shape = Initial_value_f