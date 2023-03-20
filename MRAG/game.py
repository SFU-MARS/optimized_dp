from utilities import *
from odp.Grid import Grid
from odp.solver import HJSolver, computeSpatDerivArray
from MRAG.AttackerDefender1v0 import AttackerDefender1v0
from MRAG.AttackerDefender1v1 import AttackerDefender1v1 
from MRAG.AttackerDefender2v1 import AttackerDefender2v1

# simulation 1: 4 attackers with 2 defenders
# preparations
print("Preparing for the simulaiton... \n")
T = 0.6 # total simulation time
deltat = 0.01 # calculation time interval
times = int(T/deltat)

# load all value functions, grids and spatial derivative array
value1v0 = np.load('MRAG/1v0AttackDefend.npy')
value1v1 = np.load('MRAG/1v1AttackDefend.npy')
value2v1 = np.load('MRAG/2v1AttackDefend.npy')
grid1v0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([45, 45])) # original 45
grid1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45])) # original 45
grid2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30]))
agents_1v0 = AttackerDefender1v0(uMode="min", dMode="max")
agents_1v1 = AttackerDefender1v1(uMode="min", dMode="max")  # 1v1 (4 dims dynamics)
agents_2v1 = AttackerDefender2v1(uMode="min", dMode="max")  # 2v1 (6 dim dynamics)
# 2v1 
a1x_2v1 = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=1, accuracy="low")
a1y_2v1 = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=2, accuracy="low")
a2x_2v1 = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=3, accuracy="low")
a2y_2v1 = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=4, accuracy="low")
d1x_2v1 = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=5, accuracy="low")
d1y_2v1 = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=6, accuracy="low")
# 1v1
a1x_1v1 = computeSpatDerivArray(grid1v1, value1v1, deriv_dim=1, accuracy="low")
a1y_1v1 = computeSpatDerivArray(grid1v1, value1v1, deriv_dim=2, accuracy="low")
d1x_1v1 = computeSpatDerivArray(grid1v1, value1v1, deriv_dim=3, accuracy="low")
d1y_1v1 = computeSpatDerivArray(grid1v1, value1v1, deriv_dim=4, accuracy="low")
# 1v0
x1_1v0 = computeSpatDerivArray(grid1v0, value1v0, deriv_dim=1, accuracy='low')
x2_1v0 = computeSpatDerivArray(grid1v0, value1v0, deriv_dim=2, accuracy='low')


# initialize positions of attackers and defenders
attackers_initials = [(0.0, 0.0), (0.0, 0.8), (-0.5, 0.0), (0.5, -0.5)]
defenders_initials = [(0.3, 0.5), (-0.3, 0.5)]
num_attacker = len(attackers_initials)
num_defender = len(defenders_initials)
attackers_trajectory  = [[] for _ in range(num_attacker)]
defenders_trajectory = [[] for _ in range(num_defender)]
capture_decisions = []

current_attackers = attackers_initials
current_defenders = defenders_initials

print("The simulation starts: \n")
# simulation starts
for _ in range(0, times):
    print(f"The attackers in the {_} step are at {current_attackers} \n")
    print(f"The defenders in the {_} step are at {current_defenders} \n")

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
            control_defenders.append(defender_control2(agents_2v1, joint_states2v1, a1x_2v1, a1y_2v1, a2x_2v1, a2y_2v1, d1x_2v1, d1y_2v1))
        elif len(selected[j]) == 1: # defender j capture the attacker selected[j][0]
            a1x, a1y = current_attackers[selected[j][0]]
            joint_states1v1 = (a1x, a1y, d1x, d1y)
            control_defenders.append(defender_control1(agents_1v1, joint_states1v1, a1x_1v1, a1y_1v1, d1x_1v1, d1y_1v1))
        else:  # defender j could not capture any of attackers
            attacker_index = select_attacker(d1x, d1y, current_attackers)  # choose the nearest attacker
            a1x, a1y = current_attackers[attacker_index]
            joint_states1v1 = (a1x, a1y, d1x, d1y)
            control_defenders.append(defender_control1(agents_1v1, joint_states1v1, a1x_1v1, a1y_1v1, d1x_1v1, d1y_1v1))

    print(f'The control in the {_} step of defenders are {control_defenders} \n')
    # update the next postions of defenders
    newd_positions = next_positions(current_defenders, control_defenders, deltat)
    current_defenders = newd_positions
    
    # calculate the current controls of attackers
    control_attackers = attackers_control(agents_1v0, current_attackers, x1_1v0, x2_1v0)
    print(f'The control in the {_} step of attackers are {control_attackers} \n')

    # update the next postions of attackers
    newa_positions = next_positions(current_attackers, control_attackers, deltat)
    current_attackers = newa_positions

    # document the new current positions of attackers and defenders
    for i in range(num_attacker):
        attackers_trajectory[i].append(current_attackers[i])

    for j in range(num_defender):
        defenders_trajectory[j].append(current_defenders[j])


print("The game is over.")