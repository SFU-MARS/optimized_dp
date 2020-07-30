import numpy as np
from Grid.GridProcessing import grid
from Shapes.ShapesFunctions import *

# Specify the  file that includes dynamic systems
from dynamics.Humannoid6D_sys1 import *
from dynamics.DubinsCar4D import *
from dynamics.DubinsCar import *
from dynamics.RelDyn5D import *
import scipy.io as sio

import math

""" USER INTERFACES
- Define grid

- Generate initial values for grid using shape functions

- Time length for computations

- Run
"""
# Humanoid_6D
"""
# Grid field in this order: min_range, max_range, number of dims, grid dimensions, list of periodic dim: starting at 0
g = grid(np.array([-0.5, -1.0, 0.5, -2.0, -math.pi/2, -8.0]), np.array([0.5, 1.0, 1.5, 2.0, math.pi/2, 8.0]), 6, np.array([27, 26, 27, 26, 27, 26]))

# Define my object
my_car = Humanoid_6D()

# Use the grid to initialize initial value function
Initial_value_f = Rectangle6D(g)

# Look-back length and time step
lookback_length = 2.0
t_step = 0.05

tau = np.arange(start = 0, stop = lookback_length + t_step, step = t_step)
print("Welcome to optimized_dp \n")

# Use the following variable to specify the characteristics of computation
compMethod = "minVWithVInit"
my_object  = my_car
my_shape = Initial_value_f
"""

# Dubinscar_4D

"""
g = grid(np.array([-5.0, -5.0, -1.0, -math.pi]), np.array([5.0, 5.0, 1.0, math.pi]), 4, np.array([40, 40, 50, 50]), [3])

# Define my object
my_car = DubinsCar4D()

# Use the grid to initialize initial value function
Initial_value_f = CylinderShape(g, [3,4], np.zeros(4), 1)

# Look-back length and time step
lookback_length = 2.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start = 0, stop = lookback_length + small_number, step = t_step)
print("Welcome to optimized_dp \n")

# Use the following variable to specify the characteristics of computation
# compMethod = "none"
compMethod = "minVWithVInit"
my_object  = my_car
my_shape = Initial_value_f
"""

# Relative dynamics 5D
g = grid(np.array([-10.0, -10.0, -math.pi, 0, 0]), np.array([10.0, 10.0, math.pi, 17, 17]), 5, np.array([41, 41, 31, 35, 35]), [2])

# Define my object
# my_car = RelDyn_5D(x=[0, 0, 0, 0, 0], uMin=np.array([-0.345, -5]), uMax=np.array([0.345, 3]),
#                    dMin=np.array([-math.pi / 6, -5]), dMax=np.array([math.pi / 6, 3]), dims=5, uMode="max", dMode="min")
# TODO: Driving mode 1: d1: w_h, d2: a_h
# d1: w_h in [-0.1, 0.1], d2: a_h in [-5, -2]
my_car = RelDyn_5D(x=[0, 0, 0, 0, 0], uMin=np.array([-0.345, -5]), uMax=np.array([0.345, 3]),
                   dMin=np.array([-0.1, -5]), dMax=np.array([0.1, -2]), dims=5, uMode="max", dMode="min")
# TODO: Driving mode 2: d1: w_h, d2: a_h
# d1: w_h in [-pi/6, pi/6], d2: a_h in [-2, 1]
# my_car = RelDyn_5D(x=[0, 0, 0, 0, 0], uMin=np.array([-0.345, -5]), uMax=np.array([0.345, 3]),
#                    dMin=np.array([-math.pi / 6, -2]), dMax=np.array([math.pi / 6, 1]), dims=5, uMode="max", dMode="min")
# TODO: Driving mode 3: d1: w_h, d2: a_h
# d1: w_h in [-0.2, 0.2], d2: a_h in [1, 3]
# my_car = RelDyn_5D(x=[0, 0, 0, 0, 0], uMin=np.array([-0.345, -5]), uMax=np.array([0.345, 3]),
#                    dMin=np.array([-0.2, 1]), dMax=np.array([0.2, 3]), dims=5, uMode="max", dMode="min")
print("Computing relative dynamics 5D")

# Use the grid to initialize initial value function
Initial_value_f = CylinderShape(g, [3, 4, 5], np.zeros(5), 2)

# Look-back length and time step
lookback_length = 5.2
t_step = 0.05

small_number = 1e-5
tau = np.arange(start = 0, stop = lookback_length + small_number, step = t_step)
print("Welcome to optimized_dp \n")

# Use the following variable to specify the characteristics of computation
# compMethod = "none" # Reachable set
compMethod = "minVWithVInit" # Reachable tube
my_object  = my_car
my_shape = Initial_value_f
