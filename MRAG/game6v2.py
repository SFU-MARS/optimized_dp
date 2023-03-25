from utilities import *
from odp.Grid import Grid
from compute_opt_traj import compute_opt_traj1v0
from odp.solver import HJSolver, computeSpatDerivArray
from MRAG.AttackerDefender1v0 import AttackerDefender1v0
from MRAG.AttackerDefender1v1 import AttackerDefender1v1 
from MRAG.AttackerDefender2v1 import AttackerDefender2v1
from odp.Plots.plotting_utilities import plot_simulation

# This debug for not loading spatial derivatives array before the game
# simulation 2: 6 attackers with 2 defenders
# preparations
print("Preparing for the simulaiton... \n")
T = 1.2 # total simulation time
deltat = 0.005 # calculation time interval
times = int(T/deltat)

# load all value functions, grids and spatial derivative array
value1v0 = np.load('MRAG/1v0AttackDefend.npy')  # value1v0.shape = [100, 100, len(tau)]
v1v1 = np.load('MRAG/1v1AttackDefend.npy')
value1v1 = v1v1[..., np.newaxis]  # value1v1.shape = [45, 45, 45, 45, 1]
v2v1 = np.load('MRAG/2v1AttackDefend.npy')
value2v1 = v2v1[..., np.newaxis]  # value2v1.shape = [30, 30, 30, 30, 30, 30, 1]
grid1v0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) # original 45
grid1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45])) # original 45
grid2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([36, 36, 36, 36, 36, 36]))  # 36, 36, 36, 36, 36, 36
agents_1v0 = AttackerDefender1v0(uMode="min", dMode="max")
agents_1v1 = AttackerDefender1v1(uMode="min", dMode="max")  # 1v1 (4 dims dynamics)
agents_2v1 = AttackerDefender2v1(uMode="min", dMode="max")  # 2v1 (6 dim dynamics)
tau1v0 = np.arange(start=0, stop=2.5 + 1e-5, step=0.025)
tau1v1 = np.arange(start=0, stop=4.5 + 1e-5, step=0.025)
tau2v1 = np.arange(start=0, stop=4.5 + 1e-5, step=0.025)

# initialize positions of attackers and defenders
attackers_initials = [(0.0, 0.0), (0.0, 0.8), (-0.8, 0.0), (0.5, -0.5), (-0.5, -0.3), (0.8, -0.5)]
defenders_initials = [(0.3, 0.5), (-0.3, -0.5)]  # , (0.3, -0.5)
num_attacker = len(attackers_initials)
num_defender = len(defenders_initials)
attackers_trajectory  = [[] for _ in range(num_attacker)]
defenders_trajectory = [[] for _ in range(num_defender)]
# for plotting
attackers_x = [[] for _ in range(num_attacker)]
attackers_y = [[] for _ in range(num_attacker)]
defenders_x = [[] for _ in range(num_defender)]
defenders_y = [[] for _ in range(num_defender)]
capture_decisions = []

current_attackers = attackers_initials
current_defenders = defenders_initials

controls_attacker = [[] for _ in range(num_attacker)]

# document the initial positions of attackers and defenders
for i in range(num_attacker):
    attackers_trajectory[i].append(current_attackers[i])
    attackers_x[i].append(current_attackers[i][0])
    attackers_y[i].append(current_attackers[i][1])

for j in range(num_defender):
    defenders_trajectory[j].append(current_defenders[j])
    defenders_x[j].append(current_defenders[j][0])
    defenders_y[j].append(current_defenders[j][1])

# initialize the captured results
captured_lists = []
current_captured = [0 for _ in range(num_attacker)]
captured_lists.append(current_captured)

Pc_logs = []

print("The simulation starts: \n")
# simulation starts
for _ in range(0, times):
    print(f"The attackers in the {_} step are at {current_attackers} \n")
    print(f"The defenders in the {_} step are at {current_defenders} \n")

    Ic = capture_individual(current_attackers, current_defenders, v1v1)
    Pc, value_list = capture_pair(current_attackers, current_defenders, v2v1)
    selected = mip_solver(num_attacker, num_defender, Pc, Ic)

    # Debug zone
    print("Pc is {}".format(Pc[1]))
    print("value list is {}".format(value_list[1]))
    A1x, A1y = current_attackers[2]
    A2x, A2y = current_attackers[4]
    D1x, D1y = current_defenders[1]
    A1xs, A1ys, A2xs, A2ys, D1xs, D1ys = lo2slice2v1((A1x, A1y, A2x, A2y, D1x, D1y))
    print(f"The current value of (A2, A4, D1) is {v2v1[A1xs, A1ys, A2xs, A2ys, D1xs, D1ys]}. \n")
    print(f"The current slices of (A2, A4, D1) is {A1xs, A1ys, A2xs, A2ys, D1xs, D1ys}. \n")
    print(f"The result of the MIP in the step {_} is {selected}. \n")


    Pc_logs.append(Pc)
    capture_decisions.append(selected)  # document the capture results

    # check the capture relationship
    current_captured = capture_check(current_attackers, current_defenders, selected, current_captured)
    # print("current_capture {}".format(current_captured))
    # print("captured_list {}".format(captured_lists))
    captured_lists.append(current_captured)
    # print(f"The current captured attackers are {captured}. \n")

    # calculate the current controls of defenders
    control_defenders = []  # current controls of defenders, [(d1xc, d1yc), (d2xc, d2yc)]
    for j in range(num_defender):
        d1x, d1y = current_defenders[j]
        if len(selected[j]) == 2:  # defender j capture the attacker selected[j][0] and selected[j][1]
            a1x, a1y = current_attackers[selected[j][0]]
            a2x, a2y = current_attackers[selected[j][1]]
            joint_states2v1 = (a1x, a1y, a2x, a2y, d1x, d1y)
            control_defenders.append(defender_control2v1_1slice(agents_2v1, grid2v1, value2v1, tau2v1, joint_states2v1))
        elif len(selected[j]) == 1: # defender j capture the attacker selected[j][0]
            a1x, a1y = current_attackers[selected[j][0]]
            joint_states1v1 = (a1x, a1y, d1x, d1y)
            control_defenders.append(defender_control1v1_1slice(agents_1v1, grid1v1, value1v1, tau1v1, joint_states1v1))
        else:  # defender j could not capture any of attackers
            attacker_index = select_attacker(d1x, d1y, current_attackers)  # choose the nearest attacker
            a1x, a1y = current_attackers[attacker_index]
            joint_states1v1 = (a1x, a1y, d1x, d1y)
            control_defenders.append(defender_control1v1_1slice(agents_1v1, grid1v1, value1v1, tau1v1, joint_states1v1))
  
    # print(f'The control in the {_} step of defenders are {control_defenders} \n')
    # update the next postions of defenders
    newd_positions = next_positions(current_defenders, control_defenders, deltat)  # , selected, current_captured
    current_defenders = newd_positions
    
    # calculate the current controls of attackers
    control_attackers = attackers_control(agents_1v0, grid1v0, value1v0, tau1v0, current_attackers)
    # print(f'The control in the {_} step of attackers are {control_attackers} \n')

    # update the next postions of attackers
    newa_positions = next_positions(current_attackers, control_attackers, deltat)  # , current_captured
    current_attackers = newa_positions

    # document the new current positions of attackers and defenders
    for i in range(num_attacker):
        attackers_trajectory[i].append(current_attackers[i])
        attackers_x[i].append(current_attackers[i][0])
        attackers_y[i].append(current_attackers[i][1])

    for j in range(num_defender):
        defenders_trajectory[j].append(current_defenders[j])
        defenders_x[j].append(current_defenders[j][0])
        defenders_y[j].append(current_defenders[j][1])


print("The game is over.")

print(f"The MIP results are {capture_decisions}. \n")
# print(f"The Pc logs is {Pc_logs}. \n")
print(f"The final captured_status of all attackers is {captured_lists[-1]}. \n")
# plot_simulation(attackers_x, attackers_y, [], [])
plot_simulation(attackers_x, attackers_y, defenders_x, defenders_y)
# print(f"The final positions of attackers are {attackers_trajectory[0][-1]} and {attackers_trajectory[1][-1]}. \n")
print(f"The distance between the defender and the attacker2 is {distance(defenders_trajectory[1][-1], attackers_trajectory[2][-1])}. \n")
print(f"The distance between the defender and the attacker4 is {distance(defenders_trajectory[1][-1], attackers_trajectory[4][-1])}. \n")
print(f"The captured logs is {captured_lists}. \n")
# check the distance 

# # Debug print
# v2v1_log = []

# for i in range(len(attackers_trajectory[0])):
#     a1x, a1y = attackers_trajectory[2][i]
#     a2x, a2y = attackers_trajectory[4][i]
#     d1x, d1y = defenders_trajectory[1][i]
#     jointstates2v1 = (a1x, a1y, a2x, a2y, d1x, d1y)
#     print("joint state is {}".format(jointstates2v1))
#     a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice = lo2slice2v1(jointstates2v1)

#     index = []
#     grid_points = np.linspace(-1, +1, num=36)
#     for i, s in enumerate(jointstates2v1):
#         idx = np.searchsorted(grid_points, s)
#         if idx > 0 and (
#             idx == len(grid_points)
#             or math.fabs(s - grid_points[idx - 1])
#             < math.fabs(s - grid_points[idx])
#         ):
#             index.append(idx - 1)
#         else:
#             index.append(idx)

#     print("index is {}".format(index))
#     a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice = tuple(index)
#     print("hey yo {}".format(v2v1[a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice]))
#     crap = (a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice)
#     print("joint index is {}".format(crap))
#     v2v1_log.append(v2v1[a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice])
# print(f"The 2v1 value function of the whole trajectory is {v2v1}. \n")