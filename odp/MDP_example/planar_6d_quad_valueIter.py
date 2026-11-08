import numpy as np
import math

from odp.solver import *
from odp.Grid import Grid
from itertools import product

# from odp.valueIteration import value_iteration_2D

import numpy as np
import math
from memory_profiler import memory_usage

import heterocl as hcl
import os

# 6D planar quadrotor example
g = Grid(minBounds=np.array([0., 0., -2, -2., -math.pi, -5.]),
         maxBounds=np.array([4., 4., 2., 2.,  math.pi, 5.]),
         dims=6,
         pts_each_dim=np.array([40, 40, 30, 30, 36, 10]))

thrust_control = np.linspace(0, 16, 16)
angle_accel_control = np.linspace(-20, 20, 10)
controls = np.array(list(product(thrust_control, angle_accel_control))) 
print(f"control shape is {controls.shape}")

gamma = np.array([0.99])
epsilon = np.array([1e-2])
maxIters = np.array([200])

class planar_6d_example:
    def __init__(self):
        # Some constant parameters for pendumlum adapted from the openAI gym dynamics
        self.goal_center = np.array([3., 3.]) 
        self.goal_rad= 0.75

        self.obstacle_center = np.array([1., 1.])
        self.obs_rad = 0.3
        self.dt = 0.25  # time step
        self.max_speed = 2.
        self.angular_speed = 5.0  # max angular speed
        self.maxTransitions = 1  # For this example, we assume deterministic transitions
    
    def compute_next_state(self, sVals, u):
        next_state = hcl.compute((6,), lambda *x: 0, "next_state")
        xdot = hcl.scalar(0, "xdot")
        zdot = hcl.scalar(0, "zdot")
        vx_dot = hcl.scalar(0, "vx_dot")
        vz_dot = hcl.scalar(0, "vz_dot")
        thdot = hcl.scalar(0, "thdot")
        omega_dot = hcl.scalar(0, "omega_dot")

        xdot[0] = sVals[2]  # x velocity
        zdot[0] = sVals[3]  # z velocity
        vx_dot[0] = -u[0] * hcl.sin(sVals[4])
        vz_dot[0] = u[0] * hcl.cos(sVals[4]) - 9.81  # z acceleration (gravity)
        thdot[0] = sVals[5]
        omega_dot[0] = u[1]

        next_state[0] = sVals[0] + xdot[0] * self.dt
        next_state[1] = sVals[1] + zdot[0] * self.dt
        
        next_state[2] = sVals[2] + vx_dot[0] * self.dt
        next_state[3] = sVals[3] + vz_dot[0] * self.dt
        
        # Bound the new speeds
        with hcl.if_(next_state[2] > self.max_speed):
           next_state[2] = self.max_speed
        with hcl.if_(next_state[2] < -self.max_speed):
            next_state[2] = -self.max_speed

        with hcl.if_(next_state[3] > self.max_speed):
           next_state[3] = self.max_speed
        with hcl.if_(next_state[3] < -self.max_speed):
            next_state[3] = -self.max_speed

        next_state[4] = sVals[4] + thdot[0] * self.dt
         # Normalize angles
        with hcl.if_(next_state[4] >= math.pi):
            next_state[4] = next_state[4] - 2*math.pi
        with hcl.elif_(next_state[4] < -math.pi):
            next_state[4] = next_state[4] + 2*math.pi

        next_state[5] = sVals[5] + omega_dot[0] * self.dt
        # Bound the angular velocity
        with hcl.if_(next_state[5] >= self.angular_speed):
            next_state[5] = self.angular_speed
        with hcl.elif_(next_state[5] < -self.angular_speed):
            next_state[5] = -self.angular_speed

        return next_state


    def transition(self, sVals, iVals, u):
        """
        Transition function for the planar quadrotor system.
        Inputs:    
            sVals: state values 
            iVals: indices of the state values in the grid
            u: action taken 
        Outputs:
            trans_matrix: a 2D matrix containing the possible successor states when taking u from current state sVals.
            Size: (number of transitions, 1 + number of state dimensions).
            Convention: trans_matrix[i, 0] is the probability of making transition i,
        """
        trans_matrix = hcl.compute((self.maxTransitions, (1 + 6)), lambda *x: 0, "trans_matrix")
     
        # Probability of 1 for deterministic transition 
        trans_matrix[0, 0] = 1.0
        next_state = self.compute_next_state(sVals, u)
        for k in range(6):
            trans_matrix[0, k + 1] = next_state[k]
        return trans_matrix

    # Return the reward for taking action from state
    def reward(self, sVals, iVals, u):
        """
        Reward function for the pendulum system (adopted from openAI gym).
        Inputs:
            sVals: state values [theta, theta_dot]
            iVals: indices of the state values in the grid
            u: action taken (torque applied)
        Outputs:
            rwd: the reward for taking action u from state sVals
        """

        # Variable declaration
        rwd = hcl.scalar(0, "rwd")
        dist_to_goal = hcl.scalar(0, "dist_to_goal")
        dist_to_obs_center = hcl.scalar(0, "dist_to_obs")
        
        next_state = self.compute_next_state(sVals, u)   

        dist_to_goal[0] = (next_state[0] - self.goal_center[0]) * (next_state[0] - self.goal_center[0]) +\
                        (next_state[1] - self.goal_center[1]) * (next_state[1] - self.goal_center[1])
        
        dist_to_obs_center[0] = (next_state[0] - self.obstacle_center[0]) * (next_state[0] - self.obstacle_center[0]) +\
                             (next_state[1] - self.obstacle_center[1]) * (next_state[1] - self.obstacle_center[1])
        
        with hcl.if_(dist_to_obs_center <= self.obs_rad * self.obs_rad):
            # If in the obstacle, return large negative reward
            rwd[0] = -100
        with hcl.elif_(dist_to_goal <= self.goal_rad * self.goal_rad):
            rwd[0] = +20 # Negative distance to goal as reward
        with hcl.else_():
            rwd[0] = -1
        return rwd[0]
    
# pendulum_system = planar_6d_example()
# result = solveValueIteration(planar_6d_example(),
#                              grid=g, action_space=controls,
#                              gamma=gamma, epsilon=epsilon,
#                              maxIters=maxIters
#                              )
# np.save('hcl_planar_6d_example.npy', result)
memory_usage_planar_6d, result = memory_usage((solveValueIteration, (planar_6d_example(), g, controls, gamma, epsilon, maxIters))
                                      ,retval=True, interval=0.1, max_usage=True)
print(f"Peak memory usage for planar 6D quadrotor example: {memory_usage_planar_6d} MB")
np.save('hcl_planar_6d_example_2.npy', result)
