import gymnasium as gym
import numpy as np
import math
from odp.Grid import Grid
import time

# Parameters provided by gym env
dt = 0.05
g =10
m = 1.
l = 1.
coeff1 = 3*g/(2*l)
max_speed = 8.
coeff2 = 3/(m * l * l)

grid_size=[201, 401]
angles = np.linspace(-math.pi, math.pi, grid_size[0])
angles_vel = np.linspace(-8., 8., grid_size[1])
action_list = np.linspace(-2, 2, 41)

def computeNextStates2(state, action):
    # th = math.atan2(state[1], state[0])
    th = state[0]
    if th >= math.pi:
        th -= 2 * math.pi
    elif th < -math.pi:
        th += 2 * math.pi
    reward_predicted =  -(th * th + 0.1* state[1] * state[1] +
                       0.001 *action * action)
    newthdot = state[1] + (coeff1 * math.sin(th) + coeff2 * action) * dt
    if newthdot > max_speed:
        newthdot = max_speed
    elif newthdot < -max_speed:
        newthdot = -max_speed
    new_th = th + newthdot * dt
    if new_th >= math.pi:
        new_th -= 2 * math.pi
    elif new_th < -math.pi:
        new_th += 2 * math.pi
    return (new_th, newthdot), reward_predicted

def computeNextStates(state, action):
    th = math.atan2(state[1], state[0])

    if th >= math.pi:
        th -= 2 * math.pi
    elif th < -math.pi:
        th += 2 * math.pi
    # print("theta {}".format(th))
    # print("angular speed {}".format(state[2]))

    reward_predicted =  -(th * th + 0.1* state[2] * state[2] +
                       0.001 *action * action)

    newthdot = state[2] + (coeff1 * math.sin(th) + coeff2 * action) * dt
    if newthdot > max_speed:
        newthdot = max_speed
    elif newthdot < -max_speed:
        newthdot = -max_speed
    new_th = th + newthdot * dt
    if new_th > math.pi:
        new_th -= 2 * math.pi
    elif new_th < -math.pi:
        new_th += 2 * math.pi
    return (new_th, newthdot), reward_predicted

def eval_next_state(V, state, action):
    deb, _ = computeNextStates(state, action)
    idx1  = state_to_idx(deb[0], angles)
    idx2  = state_to_idx(deb[1], angles_vel)
    return deb, V[idx1, idx2]

def state_to_idx(continous_state, state_list):
    # print(np.argmin(continous_state-state_list))
    return np.argmin(np.abs(continous_state-state_list))

def value_iter_3d(gamma=0.99
                  ,epsilon=0.05):

    V= np.zeros([grid_size[0], grid_size[1]])
    angles = np.linspace(-math.pi, math.pi, grid_size[0])
    angles_vel = np.linspace(-8., 8., grid_size[1])
    # action_list = np.linspace(-2, 2, 21)
    fill_val = -400

    error = 1e5
    epsilon = .00001
    iter_count = 0
    start = time.time()
    #while error > epsilon:
    while iter_count < 1000:
        iter_count += 1
        max_error = -1e9
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                max_val = -1e9
                for a in action_list:
                    next_state, rwd = computeNextStates2(
                        (angles[i], angles_vel[j]), a)
                    i1 = state_to_idx(next_state[0], angles)
                    i2 = state_to_idx(next_state[1], angles_vel)
                    if (rwd + gamma * V[i1, i2]) > max_val:
                        max_val = (rwd + gamma * V[i1, i2])
                max_error = max(abs(max_val-V[i, j ]), max_error)
                V[i, j] = max_val
        if max_error < epsilon:
            break
        error = max_error
        print("At iter {} with error {}".format(iter_count, error))
    print("It took {} iterations and {} seconds to converge with error"
          " {}".format(iter_count,
                       (time.time() - start),  error))
    np.save("pendulum_python_gamma08_new.npy", V)
    return V



env = gym.make("Pendulum-v1", g=10, render_mode="human")

def set_initial_state(env, theta, theta_dot):
    env.reset()  # Reset the environment
    env.state = np.array([theta, theta_dot])  # Manually set the state
    return env

# env = set_initial_state(env, 0., 0.)
obs, info = env.reset(seed=30)

# Load if computed from odp
V = np.load("hcl_pendulum_res_new.npy")

# action_list = np.linspace(-2, 2, 11)
best_state = (-1., 0, 0.)

rewards_list = []
rew = 0
for j in range(2000):
    # Should have used Q-learning instead
    max_val=-1e9

    best_a = action_list[0]
    state_for_best_a = []
    for a in action_list:
        new_state, Vs_tp1 = eval_next_state(V, obs, a)
        if Vs_tp1 > max_val:
            best_a = a
            max_val = Vs_tp1
            state_for_best_a = new_state

    obs, r, terminated, trunc, inf = env.step([best_a])
    rew += r
    if terminated or trunc:
        obs, inf = env.reset()
        rewards_list.append(rew)
        #print(rew)
        rew = 0
print(rewards_list)
env.close()

