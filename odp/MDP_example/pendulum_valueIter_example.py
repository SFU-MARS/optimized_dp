import numpy as np
import math

from odp.solver import *
from odp.Grid import Grid
# from odp.valueIteration import value_iteration_2D

import numpy as np
import math
import heterocl as hcl
import os

bounds = np.array([[-math.pi, math.pi], [-8., 8.]])
ptsEachDim = np.array([201, 401])

g = Grid(minBounds=np.array([-math.pi, -8.]),
         maxBounds=np.array([math.pi, 8.]),
         dims=2,
         pts_each_dim=np.array([201, 401]))

# Set goal to b
torques = np.linspace(-2., 2., 41)
gamma = np.array([0.99])
epsilon = np.array([1.117e-5])
maxIters = np.array([1500])

# Deterministic case - dynamics based on pendulum dynamics
num_dims = 2 # number of dimensions

class pendulum_2d_example:
    def __init__(self):
        # Some constant parameters for pendumlum adapted from the openAI gym dynamics
        self.dt = 0.05
        self.g = 10
        self.m = 1.
        self.l = 1.
        self.max_speed = 8.
        self.coeff1 = 3 * self.g/ (2* self.l)
        self.coeff2 = 3.0/(self.m * self.l * self.l)
        self.maxTransitions = 1 

    def transition(self, sVals, iVals, u):
        """
        Transition function for the pendulum system (adopted from ).
        Inputs:    
            sVals: state values [theta, theta_dot]
            iVals: indices of the state values in the grid
            u: action taken (torque applied)
        Outputs:
            trans_matrix: a 2D matrix containing the possible successor states when taking u from current state sVals.
            Size: (number of transitions, 1 + number of state dimensions).
            Convention: trans_matrix[i, 0] is the probability of making transition i,
        """

        trans_matrix = hcl.compute((self.maxTransitions, (1 + 2)), lambda *x: 0, "trans_matrix")
        # Variable declaration
        newthdot = hcl.scalar(0, "newthdot")
        th = hcl.scalar(0, "th")
        new_th = hcl.scalar(0, "new_th")

        # Just use theta from goals variable
        th[0] = sVals[0]

        newthdot[0] = sVals[1] + (self.coeff1 * hcl.sin(sVals[0]) +  self.coeff2 * u) * self.dt
        
        # Bound the new speed
        with hcl.if_(newthdot[0] > self.max_speed):
            newthdot[0] = self.max_speed
        with hcl.if_(newthdot[0] < -self.max_speed):
            newthdot[0] = -self.max_speed
        new_th[0] = th[0] + newthdot[0] * self.dt

        # Normalize angles
        with hcl.if_(new_th[0] >= math.pi):
            new_th[0] = new_th[0] - 2*math.pi
        with hcl.elif_(new_th[0] < -math.pi):
            new_th[0] = new_th[0] + 2*math.pi
        
        # Probability of 1 for deterministic transition 
        trans_matrix[0, 0] = 1.0
        trans_matrix[0, 1] = new_th[0]
        trans_matrix[0, 2] = newthdot[0]

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
        rwd[0] = -(sVals[0] * sVals[0] + 0.1 * sVals[1] * sVals[1] + 0.001 * u * u)
        return rwd[0]


pendulum_system = pendulum_2d_example()
result = solveValueIteration(pendulum_system,
                             grid=g, action_space=torques,
                             gamma=gamma, epsilon=epsilon,
                             maxIters=maxIters
                             )
np.save('hcl_pendulum_res_new3.npy', result)