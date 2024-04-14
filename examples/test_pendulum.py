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

def computeNextStates2(state, action):
    # th = math.atan2(state[1], state[0])
    th = state[0]
    # print("theta {}".format(th))
    # print("angular speed {}".format(state[2]))

    reward_predicted =  -(th * th + 0.1* state[1] * state[1] +
                       0.001 *action * action)

    newthdot = state[1] + (coeff1 * math.sin(th) + coeff2 * action) * dt
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

def computeNextStates(state, action):
    th = math.atan2(state[1], state[0])
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

def eval_next_state(g, V, state, action):
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
    return g.get_value(V, deb)

def state_to_idx(continous_state, state_list):
    # print(np.argmin(continous_state-state_list))
    return np.argmin(continous_state-state_list)

def value_iter_3d(grid_size=[100, 100], gamma=0.9
                  ,epsilon=0.05):

    V= np.zeros([grid_size[0], grid_size[1]])
    angles = np.linspace(-math.pi, math.pi, grid_size[0])
    angles_vel = np.linspace(-8., 8., grid_size[1])
    action_list = np.linspace(-2, 2, 100)
    fill_val = -400

    error = 1e5
    iter_count = 0
    start = time.time()
    #while error > epsilon:
    while iter_count <= 60:
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
        error = max_error
        print("At iter {} with error {}".format(iter_count, error))
    print("It took {} iterations and {} mins to converge with error"
          " {}".format(iter_count,
                       (time.time() - start)/60,  error))
    return V


env = gym.make("Pendulum-v1", g=10, render_mode="human")
obs, info = env.reset(seed=41)

teta = math.atan2(obs[1], obs[0])
print("Initial state {}".format((teta, obs[2])))
# Load if computed from odp
# V = np.load("pendulum.npy")

# DEBUG: Compute V right in ther
# V = value_iter_3d()
# np.save("new_pendulum.npy", V)
V= np.load("new_pendulum.npy")
print(np.max(V))
print(V.shape)
print(V[50, 50])
grid = Grid(np.array([-math.pi, -8]),np.array([math.pi, 8.]),
           2, np.array([V.shape[0], V.shape[1]]))
# print(grid.get_value(V, (-1.57, obs[2])))
# print(grid.get_value(V, (1.57, obs[2])))

action_list = np.linspace(-2, 2, 100)
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
for j in range(500):
    # Should have used Q-learning instead
    max_val=-1e9

        #### DEBUG TRANSITION MODEL ####
        # Pick one sample action
        # sample_a = 2.
        # next_state_predicted, rew_predicted = (
        #     computeNextStates(obs, sample_a))
        # print("Position {}".format((obs[0], obs[1])))

    best_a = action_list[0]
    # print("current state {}".format(obs))
    for a in action_list:
        Vs_tp1 = eval_next_state(grid, V, obs, a)
        if Vs_tp1 > max_val:
            best_a = a
            max_val = Vs_tp1
        #print("It's vaalue {}".format(Vs_tp1))
    # print("chosen max_val {}".format(max_val))
    # print(obs[2])

    obs, r, terminated, trunc, inf = env.step([best_a])
        # obs, r, terminated, trunc, inf = env.step([sample_a])

        # # Compare the two states
        # theta = math.atan2(obs[1], obs[0])
        # print("Iterate {}: predicted_state {}, observed_state {}, "
        #       "predicted reward {}, observed reward {}".format(j,
        #                     next_state_predicted, (theta, obs[2]),
        #                     rew_predicted, r))

    rew += r
    #print("here")
    #env.render()
    if terminated or trunc:
        #print("here")
        obs, inf = env.reset()
        rewards_list.append(rew)
        #print(rew)
        rew = 0

env.close()

