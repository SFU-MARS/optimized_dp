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
    # print("theta {}".format(th))
    # print("angular speed {}".format(state[2]))
    reward_predicted =  -(th * th + 0.1* state[1] * state[1] +
                       0.001 *action * action)
    #if th == 0: #and state[1] == 0 and action == 0:
    #    print("reward predictied {}".format(reward_predicted))

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
    # Just use theta from goals variable
    # th = math.atan2(state[1], state[0])
    # # print("theta {}".format(th))
    # # print("angular speed {}".format(state[2]))
    #
    # newthdot = state[2] + (coeff1 * math.sin(th) + coeff2 * action) * dt
    # if newthdot > max_speed:
    #     newthdot = max_speed
    # elif newthdot < -max_speed:
    #     newthdot = -max_speed
    # new_th = th + newthdot * dt
    # if new_th > math.pi:
    #     new_th -= 2*math.pi
    # elif new_th < -math.pi:
    #     new_th += 2*math.pi
    # next_state = (new_th, newthdot)
    #print("action {} and next state {}".format(action, next_state))
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
# obs = (1, 0, 0)
teta = math.atan2(obs[1], obs[0])
print("Initial state {}".format((teta, obs[2])))

# obs, r, terminated, trunc, inf = env.step([0.])
# print(obs)
# Load if computed from odp
V = np.load("hcl_pendulum_res_new.npy")
# print(np.shape(V))

# DEBUG: Compute V right in ther
# V = np.load("pendulum_python_gamma99_new.npy")


# V = value_iter_3d()
# print(V[40, :])
# print(V[0, :])
# print(np.max(np.abs(V[40, :] - V[0, :])))
# np.save("new_pendulum.npy", V)
# V= np.load("new_pendulum.npy")
print("max val is {}".format(np.max(V)))
print("min val is {}".format(np.min(V)))
# print(V[50, 50])

#grid = Grid(np.array([-math.pi, -8]),np.array([math.pi, 8.]),
#           2, np.array([V.shape[0], V.shape[1]]), [1])

# print(grid.get_value(V, (-1.57, obs[2])))
# print(grid.get_value(V, (1.57, obs[2])))

# action_list = np.linspace(-2, 2, 11)
best_state = (-1., 0, 0.)

# max_val = -1e9
# best_a= []
# for a in action_list:
#         Vs_tp1 = eval_next_state(grid, V, best_state, a)
#         print("Action {} has val {}".format(a, Vs_tp1))
#         if Vs_tp1 > max_val:
#             best_a = a
#             max_val = Vs_tp1
#         #print("It's vaalue {}".format
# print("at top action is {}".format(best_a))



rewards_list = []
rew = 0
for j in range(2000):
    # Should have used Q-learning instead
    max_val=-1e9

    best_a = action_list[0]
    state_for_best_a = []
    # print("current state {}".format(obs))
    for a in action_list:
        new_state, Vs_tp1 = eval_next_state(V, obs, a)
        if Vs_tp1 > max_val:
            best_a = a
            max_val = Vs_tp1
            state_for_best_a = new_state

    obs, r, terminated, trunc, inf = env.step([best_a])
    # print(obs)
    rew += r
    #print("here")
    #env.render()
    if terminated or trunc:
        print("here")
        obs, inf = env.reset()
        rewards_list.append(rew)
        #print(rew)
        rew = 0
print(rewards_list)
env.close()

