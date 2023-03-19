from utilities import *
from odp.Grid import Grid
from odp.solver import HJSolver, computeSpatDerivArray
from MRAG.AttackerDefender1v0 import AttackerDefender1v0
from MRAG.AttackerDefender1v1 import AttackerDefender1v1 
from MRAG.AttackerDefender2v1 import AttackerDefender2v1


# simulation 1: 4 attackers with 2 defenders
# preparations
T = 0.6 # total simulation time
deltat = 0.01 # calculation time interval

# load all value functions and grids
value1v0 = np.load('MRAG/1v0AttackDefend.npy')
value1v1 = np.load('MRAG/1v1AttackDefend.npy')
value2v1 = np.load('MRAG/2v1AttackDefend.npy')
grid1v0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([45, 45])) # original 45
grid1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45])) # original 45
grid2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30]))
agents_1v0 = AttackerDefender1v0(uMode="min", dMode="max")
agents_1v1 = AttackerDefender1v1(uMode="min", dMode="max")  # 1v1 (4 dims dynamics)
agents_2v1 = AttackerDefender2v1(uMode="min", dMode="max")  # 2v1 (6 dim dynamics)


# initialize positions of attackers and defenders
attackers_initials = [(0.0, 0.0), (0.0, 0.8), (-0.5, 0.0), (0.5, -0.5)]
defenders_initials = [(0.3, 0.5), (-0.3, 0.5)]
num_attacker = len(attackers_initials)
num_defender = len(defenders_initials)
attackers_trajectory  = [[] for _ in range(num_attacker)]
defenders_trajectory = [[] for _ in range(num_defender)]
capture_decisions = []

# simulation begins
current_attackers = attackers_initials
current_defenders = defenders_initials

# document the initial positions of attackers and defenders
for i in range(num_attacker):
    attackers_trajectory[i].append(current_attackers[i])

for j in range(num_defender):
    defenders_trajectory[j].append(current_defenders[j])

# for _ in range(0, 2, 1): in every time step
Ic = capture_individual(current_attackers, current_defenders, value1v1)
Pc = capture_pair(current_attackers, current_defenders, value2v1)
selected = mip_solver(num_attacker, num_defender, Pc, Ic)

capture_decisions.append(selected)  # document the capture results

# calculate the current controls of defenders
control_defenders = []  # current controls of defenders, [(d1xc, d1yc), (d2xc, d2yc)]
for j in range(num_defender):
    d1x, d1y = current_defenders[j]
    if len(selected[j]) == 2:  # defender j capture the attacker selected[j][0] and selected[j][1]
        a1x, a1y = current_attackers[selected[j][0]]
        a2x, a2y = current_attackers[selected[j][1]]
        joint_states2v1 = (a1x, a1y, a2x, a2y, d1x, d1y)
        control_defenders.append(defender_control2(grid2v1, value2v1, agents_2v1, joint_states2v1))
    elif len(selected[j]) == 1: # defender j capture the attacker selected[j][0]
        a1x, a1y = current_attackers[selected[j][0]]
        joint_states1v1 = (a1x, a1y, d1x, d1y)
        control_defenders.append(defender_control1(grid1v1, value1v1, agents_1v1, joint_states1v1))
    else:  # defender j could not capture any of attackers
        attacker_index = select_attacker(d1x, d1y, current_attackers)  # choose the nearest attacker
        a1x, a1y = current_attackers[attacker_index]
        joint_states1v1 = (a1x, a1y, d1x, d1y)
        control_defenders.append(defender_control1(grid1v1, value1v1, agents_1v1, joint_states1v1))

# calculate the current controls of attackers
control_attackers = attackers_control(grid1v0, value1v0, agents_1v0, current_attackers)
# print(f'The control of attackers are {control_attackers} \n')

# update the next postions of defenders
newd_positions = next_positions(current_defenders, control_defenders)
current_defenders = newd_positions

# update the next postions of attackers
newa_positions = next_positions(current_attackers, control_attackers)
current_attackers = newa_positions

# document the new current positions of attackers and defenders
for i in range(num_attacker):
    attackers_trajectory[i].append(current_attackers[i])

for j in range(num_defender):
    defenders_trajectory[j].append(current_defenders[j])






