import numpy as np
from Grid.GridProcessing import grid
from Shapes.ShapesFunctions import *

# Specify the  file that includes dynamic systems
from dynamics.DubinsCar import *
from dynamics.DubinsCar4D import *
from dynamics.DubinsCar5DAvoid import *
import scipy.io as sio

import math

""" USER INTERFACES
- Define grid

- Generate initial values for grid using shape functions

- Time length for computations

- Run
"""

# Grid field in this order: min_range, max_range, number of dims, grid dimensions, list of periodic dim: starting at 0

# Humanoid_6D
"""g = grid(np.array([-0.5, -1.0, 0.5, -2.0, -math.pi/2, -8.0]), np.array([0.5, 1.0, 1.5, 2.0, math.pi/2, 8.0]), 6, np.array([27, 26, 27, 26, 27, 26]))

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
my_shape = Initial_value_f """

#DubinsCar4D
"""
g = grid(np.array([-5.0, -5.0, -1.0, -math.pi]), np.array([5.0, 5.0, 1.0, math.pi]), 4, np.array([40, 40, 50, 50]), [3])

# Define my object
my_car = DubinsCar4D()

# Use the grid to initialize initial value function
Initial_value_f = CylinderShape(g, [3,4], np.zeros(4), 1)

# Look-back lenght and time step
lookback_length = 2.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start = 0, stop = lookback_length + small_number, step = t_step)
print("Welcome to optimized_dp \n")

# Use the following variable to specify the characteristics of computation
compMethod = "none"
my_object  = my_car
my_shape = Initial_value_f
"""

# DubinsCar 3D system
"""
g = grid(np.array([-5.0, -5.0, -math.pi]), np.array([5.0, 5.0, math.pi]), 3, np.array([80, 80, 80]), [2])
=======
g = grid(np.array([-5.0, -5.0, -math.pi]), np.array([5.0, 5.0, math.pi]), 3, np.array([80, 80, 80]), [2])

# Define my object
my_car = DubinsCar()

# Use the grid to initialize initial value function
Initial_value_f = CylinderShape(g, [3], np.zeros(3), 1)

# Look-back lenght and time step
lookback_length = 1.0
t_step = 0.02

small_number = 1e-5
tau = np.arange(start = 0, stop = lookback_length + small_number, step = t_step)
print("Welcome to optimized_dp \n")

# Define your constraint_values array here and set constrainedDefined = True

# Use the following variable to specify the characteristics of computation
compMethod = ['minVWithV0', 'minVWithCStraint']
my_object  = my_car
constrainedDefined = False
my_shape = Initial_value_f"""

# 5D Dubins Avoidance

#g = grid(np.array([-20.0, -20.0, -math.pi, -10, -10]), np.array([20, 20, math.pi, 10, 10]), 5, np.array([40, 40, 20, 20, 20]), [2])
g = grid(np.array([-20.0, -20.0, -math.pi, -5, -5]), np.array([20, 20, math.pi, 10, 10]), 5, np.array([40, 40, 20, 20, 20]), [2])

# Define my object
#my_object = DubinsCar5DAvoid(x=[0,0,0,0,0], u_theta_max = math.pi/10, u_v_max=3, d_theta_max=math.pi/10, d_v_max=3, uMode="max", dMode="min")
my_object = DubinsCar5DAvoid(x=[0,0,0,0,0], u_theta_max = math.pi/2, u_v_max=3, d_theta_max=math.pi/2, d_v_max=3, uMode="max", dMode="min")

# Use the grid to initialize initial value function

# J: Evader losing conditions
# include half plane x4 < vmin as a losing condition
# for union of the two sets use min()
# mode: minVWithV0

Initial_value_f = np.minimum(CylinderShape(g, [3,4,5], np.zeros(5), 2), HalfPlane(g, 0.1, 3))


# J: for vehicle number 2 (pursuer)
# define a set x5 > vmin it is G (constraint_values)
# mode:maxVWithCStraint
constraint_values = -HalfPlane(g, 0.1, 4)

# Look-back lenght and time step
lookback_length = 10.0
t_step = 0.05

tau = np.arange(start = 0, stop = lookback_length + t_step, step = t_step)
print("Welcome to optimized_dp \n")

# Use the following variable to specify the characteristics of computation
#compMethod = "none"
#compMethod = "minVWithVInit"
#compMethod = "minVWithV0"
#compMethod = ["minVWithV0","maxVWithCStraint"] #Juan
compMethod = ['minVWithV0', 'minVWithCStraint'] 
my_object  = my_object
my_shape = Initial_value_f
