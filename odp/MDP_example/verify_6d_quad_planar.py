import numpy as np
import math
from odp.Grid import Grid
from itertools import product

import time
import matplotlib.pyplot as plt


# 6D planar quadrotor example
g = Grid(minBounds=np.array([0., 0., -2, -2., -math.pi, -5.]),
         maxBounds=np.array([4., 4., 2., 2.,  math.pi, 5.]),
         pts_each_dim=np.array([40, 40, 15, 15, 36, 10]))
dt = 0.25  # time step
max_speed = 2.
angular_speed = 5.0  # max angular speed
goal_center = np.array([3., 3.])
goal_rad = 0.75
obstacle_center = np.array([1., 1.])
obs_rad = 0.3

thrust_control = np.linspace(0, 16, 16)
angle_accel_control = np.linspace(-20, 20, 10)
controls = np.array(list(product(thrust_control, angle_accel_control))) 


# Load if computed from odp
V = np.load("hcl_planar_6d_example.npy")


def compute_next_state_numpy(sVals, u):
    next_state = np.zeros(6)
    xdot = sVals[2]  # x velocity
    zdot = sVals[3]  # z velocity
    vx_dot = -u[0] * np.sin(sVals[4])
    vz_dot = u[0] * np.cos(sVals[4]) - 9.81  # z acceleration (gravity)
    thdot = sVals[5]
    omega_dot = u[1]    
    next_state[0] = sVals[0] + xdot * dt
    next_state[1] = sVals[1] + zdot * dt
    next_state[2] = sVals[2] + vx_dot * dt
    next_state[3] = sVals[3] + vz_dot * dt     

    # Bound the new speeds
    next_state[2] = np.clip(next_state[2], -max_speed, max_speed)
    next_state[3] = np.clip(next_state[3], -max_speed, max_speed)         
    next_state[4] = sVals[4] + thdot * dt
    # Normalize angles
    next_state[4] = next_state[4] - 2*np.pi if next_state[4] >= np.pi else next_state[4]
    next_state[4] = next_state[4] + 2*np.pi if next_state[4] < -np.pi else next_state[4]
    next_state[5] = sVals[5] + omega_dot * dt
    # Bound the angular velocity
    next_state[5] = np.clip(next_state[5], -angular_speed, angular_speed) 
    return next_state

def reward_numpy(sVals, u):
    """
    Reward function for the planar quadrotor system.
    Inputs:
        sVals: state values [x, z, xdot, zdot, theta, omega]
        u: action taken (thrust and angular acceleration)
    Outputs:
        rwd: the reward for taking action u from state sVals
    """
    dist_to_goal = np.linalg.norm(sVals[:2] - goal_center)
    dist_to_obs_center = np.linalg.norm(sVals[:2] - obstacle_center)

    if dist_to_obs_center <= obs_rad:
        return -100
    elif dist_to_goal <= goal_rad:
        return 20
    else:
        return -1

def eval_next_state(V, state, action):
    deb = compute_next_state_numpy(state, action)
    # print("state ", state)
    idx1  = state_to_idx(deb[0], g.grid_points[0])
    idx2  = state_to_idx(deb[1], g.grid_points[1])
    idx3  = state_to_idx(deb[2], g.grid_points[2])
    idx4  = state_to_idx(deb[3], g.grid_points[3])
    idx5  = state_to_idx(deb[4], g.grid_points[4])
    idx6  = state_to_idx(deb[5], g.grid_points[5])
    return deb, V[idx1, idx2, idx3, idx4, idx5, idx6], (idx1, idx2, idx3, idx4, idx5, idx6)

def state_to_idx(continous_state, state_list):
    # print(np.argmin(continous_state-state_list))
    return np.argmin(np.abs(continous_state-state_list))

# current_state = [3., 2.5, -.5, 0.5, math.pi/4, -2.]
# current_state = [2., 2., -.5, 0.5, math.pi/4, -2.]
# current_state = [2., 2., 0., 0., math.pi/4, -2.]
current_state = [0.5, 0.5, 0.0, 0.5, 0., -2.]


states_observed = []
rewards_list = []
rew = 0

for j in range(30):
    # Should have used Q-learning instead
    max_val=-1e9

    best_a = controls[0]
    state_for_best_a = []
    for a in controls:
        new_state, Vs_tp1, indices = eval_next_state(V, current_state, a)
        # print("action:", a, "Vs_tp1:", Vs_tp1, "new_state is: ", indices, "and ", new_state)
        if Vs_tp1 > max_val:
            best_a = a
            max_val = Vs_tp1
            state_for_best_a = new_state

    # Update reward after taking the best action
    rew += reward_numpy(current_state, best_a)
    print("current state:", current_state, "action:", best_a, "reward:", rew)

    current_state = state_for_best_a
    states_observed.append(current_state)


    
print(rewards_list)
# print(f"Total reward: {rew}")
# print(f"States observed: {states_observed}")

# plot x-y trajectory
x_vals = [state[0] for state in states_observed]
y_vals = [state[1] for state in states_observed]
plt.plot(x_vals, y_vals, marker='o')
plt.title("6D Planar Quadrotor Trajectory")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid()
plt.axis('equal')
plt.savefig("my_trajectory.png")
# plt.show()  


