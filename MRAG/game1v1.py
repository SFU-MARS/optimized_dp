from odp.Plots.plotting_utilities import *
from MRAG.utilities import *
from odp.Grid import Grid
from MRAG.compute_opt_traj import compute_opt_traj1v0
from odp.solver import HJSolver, computeSpatDerivArray
from copy import deepcopy
from MRAG.AttackerDefender1v0 import AttackerDefender1v0
from MRAG.AttackerDefender1v1 import AttackerDefender1v1 
from MRAG.AttackerDefender2v1 import AttackerDefender2v1
from MRAG.AttackerDefender1v2 import AttackerDefender1v2



# Simulation: 1 attacker with 2 defenders
# preparations
print("Preparing for the simulaiton... \n")
T = 0.9 # attackers_stop_times = [0.475s (95 A1 is captured), 0.69s (138 A0 by D0)]
deltat = 0.005 # calculation time interval
times = int(T/deltat)

# load all value functions, grids and spatial derivative array
value1v0 = np.load('MRAG/1v0AttackDefend.npy')  # value1v0.shape = [100, 100, len(tau)]
# # print(value1v0.shape)
v1v1 = np.load('MRAG/1v1AttackDefend_speed15.npy')
print(f"The shape of the 1v1 value function is {v1v1.shape}. \n")
# # v1v1 = np.load('MRAG/1v1AttackDefend.npy')
value1v1 = v1v1[..., np.newaxis]  # value1v1.shape = [45, 45, 45, 45, 1]
# # v2v1 = np.load('MRAG/2v1AttackDefend.npy')
# # v2v1 = np.load('2v1AttackDefend_speed15.npy') # grid = 30
# v2v1 = np.load('MRAG/2v1AttackDefend_speed15.npy')
# print(f"The shape of the 2v1 value function is {v2v1.shape}. \n")
# value2v1 = v2v1[..., np.newaxis]  # value2v1.shape = [30, 30, 30, 30, 30, 30, 1]
#TODO: Hanyang: check why
# v1v2 = np.load('MRAG/1v2AttackDefend_speed15.npy')
# print(f"The shape of the 1v2 value function is {v1v2.shape}. \n")
# value1v2 = v1v2[..., np.newaxis]  # value1v2.shape = [30, 30, 30, 30, 30, 30, 1]

grid1v0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) # original 45
grid1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45])) # original 45
# grid1v2 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30])) # original 45
# grid2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30])) # [36, 36, 36, 36, 36, 36] [30, 30, 30, 30, 30, 30]
agents_1v0 = AttackerDefender1v0(uMode="min", dMode="max")
agents_1v1 = AttackerDefender1v1(uMode="min", dMode="max")  # 1v1 (4 dims dynamics)
# agents_1v2 = AttackerDefender1v2(uMode="min", dMode="max")  # 1v2 (6 dim dynamics)
# agents_2v1 = AttackerDefender2v1(uMode="min", dMode="max")  # 2v1 (6 dim dynamics)
tau1v0 = np.arange(start=0, stop=2.5 + 1e-5, step=0.025)
tau1v1 = np.arange(start=0, stop=4.5 + 1e-5, step=0.025)
# tau1v2 = np.arange(start=0, stop=4.5 + 1e-5, step=0.025)
# tau2v1 = np.arange(start=0, stop=4.5 + 1e-5, step=0.025)

# initialize positions of attackers and defenders
attackers_initials =[(-0.15, 0.5)]  # [(0.0, 0.0), (0.0, 0.8)], [(-0.5, 0.0), (0.0, 0.8)],  [(-0.5, 0.5), (-0.3, -0.8)] [(-0.5, -0.3), (0.8, -0.5)], 
defenders_initials = [(-0.15, 0.7)]   #  [(-0.5, 0.0), (0.0, 0.8)]  [(-0.6, 0.8), (-0.6, -0.8)]

ax = attackers_initials[0][0]
ay = attackers_initials[0][1]
d1x = defenders_initials[0][0]
d1y = defenders_initials[0][1]
# d2x = defenders_initials[1][0]
# d2y = defenders_initials[1][1]

# plot 1v1 reach-avoid tube
jointstates1v1 = (ax, ay, d1x, d1y)
ax_slice, ay_slice, d1x_slice, d1y_slice = lo2slice1v1(jointstates1v1, slices=30)
value_function1v1 = value1v1[ax_slice, ay_slice, d1x_slice, d1y_slice]
print(f"The initial value function of 1vs1 is {value_function1v1}. \n")


num_attacker = len(attackers_initials)
num_defender = len(defenders_initials)
attackers_trajectory  = [[] for _ in range(num_attacker)]
defenders_trajectory = [[] for _ in range(num_defender)]

# mip results 
capture_decisions = []

# load the initial states
current_attackers = attackers_initials
current_defenders = defenders_initials
for i in range(num_attacker):
    attackers_trajectory[i].append(current_attackers[i])

for j in range(num_defender):
    defenders_trajectory[j].append(current_defenders[j])

# initialize the captured results
attackers_status_logs = []
attackers_status = [0 for _ in range(num_attacker)]
stops_index = []  # the list stores the indexes of attackers that has been captured or arrived
attackers_status_logs.append(deepcopy(attackers_status))

# # log the attackers be assigned defenders
# attacker_assigneds = []

RA1v1s = []
# RA1v2s = []

print("The simulation starts: \n")
# simulation starts
for _ in range(0, times):

    RA1v1 = capture_1vs1(current_attackers, current_defenders, v1v1, stops_index)  # attacker will win the 1 vs. 1 game
    # RA1v2 = capture_1vs2(current_attackers, current_defenders, v1v2)  # attacker will win the 1 vs. 2 game
    RA1v1s.append(RA1v1)
    # RA1v2s.append(RA1v2)

    a1x, a1y = current_attackers[0]
    d1x, d1y = current_defenders[0]
    control_defenders = [[] for num in range(num_defender)]  # current controls of defenders, [(d1xc, d1yc), (d2xc, d2yc)] 
    joint_states1v1 = (a1x, a1y, d1x, d1y)
    control_defenders[0].append(defender_control1v1_1slice(agents_1v1, grid1v1, value1v1, tau1v1, joint_states1v1))

    # update the next postions of defenders
    # newd_positions = next_positions(current_defenders, control_defenders, deltat)  # , selected, current_captured
    newd_positions = next_positions_d(current_defenders, control_defenders, deltat)
    current_defenders = newd_positions
    
    # calculate the current controls of attackers
    control_attackers = attackers_control(agents_1v0, grid1v0, value1v0, tau1v0, current_attackers)

    # update the next postions of attackers
    newa_positions = next_positions_a2(current_attackers, control_attackers, deltat, stops_index)  # , current_captured
    current_attackers = newa_positions

    # document the new current positions of attackers and defenders
    for i in range(num_attacker):
        attackers_trajectory[i].append(current_attackers[i])
       

    for j in range(num_defender):
        defenders_trajectory[j].append(current_defenders[j])


    # # check the attackers status: captured or not  
    selected = [[0]]
    attackers_status = capture_check(current_attackers, current_defenders, selected, attackers_status)
    attackers_status_logs.append(deepcopy(attackers_status))
    attackers_arrived = arrived_check(current_attackers)
    stops_index = stoped_check(attackers_status, attackers_arrived)
    print(f"The current status at iteration{_} of attackers is arrived:{attackers_arrived} + been captured:{attackers_status}. \n")

    if len(stops_index) == num_attacker:
        print(f"All attackers have arrived or been captured at the time t={(_+1)*deltat}. \n")
        break

print("The game is over. \n")

print(f"The results of the selected is {capture_decisions}. \n")
print(f"The final captured_status of all attackers is {attackers_status_logs[-1]}. \n")


print(f"The RA1v1s is {RA1v1s}. \n")
# print(f"The RA1v2s is {RA1v2s}. \n")

# Play the animation
animation_2v1(attackers_trajectory, defenders_trajectory, attackers_status_logs, T)
