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
# print(value1v0.shape)
v1v1 = np.load('MRAG/1v1AttackDefend_speed15.npy')
# v1v1 = np.load('MRAG/1v1AttackDefend.npy')
value1v1 = v1v1[..., np.newaxis]  # value1v1.shape = [45, 45, 45, 45, 1]
# v2v1 = np.load('MRAG/2v1AttackDefend.npy')
# v2v1 = np.load('2v1AttackDefend_speed15.npy') # grid = 30
v2v1 = np.load('MRAG/2v1AttackDefend_speed15.npy')
print(f"The shape of the 2v1 value function is {v2v1.shape}. \n")
value2v1 = v2v1[..., np.newaxis]  # value2v1.shape = [30, 30, 30, 30, 30, 30, 1]
#TODO: Hanyang: check why
v1v2 = np.load('MRAG/1v2AttackDefend_speed15.npy')
value1v2 = v1v2[..., np.newaxis]  # value1v2.shape = [30, 30, 30, 30, 30, 30, 1]

grid1v0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) # original 45
grid1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45])) # original 45
grid1v2 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30])) # original 45
grid2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30])) # [36, 36, 36, 36, 36, 36] [30, 30, 30, 30, 30, 30]
agents_1v0 = AttackerDefender1v0(uMode="min", dMode="max")
agents_1v1 = AttackerDefender1v1(uMode="min", dMode="max")  # 1v1 (4 dims dynamics)
agents_1v2 = AttackerDefender1v2(uMode="min", dMode="max")  # 1v2 (6 dim dynamics)
agents_2v1 = AttackerDefender2v1(uMode="min", dMode="max")  # 2v1 (6 dim dynamics)
tau1v0 = np.arange(start=0, stop=2.5 + 1e-5, step=0.025)
tau1v1 = np.arange(start=0, stop=4.5 + 1e-5, step=0.025)
tau1v2 = np.arange(start=0, stop=4.5 + 1e-5, step=0.025)
tau2v1 = np.arange(start=0, stop=4.5 + 1e-5, step=0.025)

# initialize positions of attackers and defenders
# ## not work:
# attackers_initials =[(-0.2, 0.0)]  # [(-0.2, 0.0)] 
# defenders_initials = [(-0.8, 0.5), (-0.8, -0.5)]   #  [(-0.8, 0.5), (-0.8, -0.5)]
# work but not due to 1vs2
# attackers_initials =[(-0.15, 0.5)]  # [(-0.15, 0.5)]
# defenders_initials = [(-0.5, 0.5), (0.0, 0.2)]  # [(-0.5, 0.5), (0.0, 0.2)]

attackers_initials = [(-0.4, 0.0)] # [(-0.5, 0.0), (0.0, 0.8)]
defenders_initials = [(-0.8, -0.8), (-0.8, 0.3)]  #  [(-0.8, -0.3)]

num_attacker = len(attackers_initials)
num_defender = len(defenders_initials)
attackers_trajectory  = [[] for _ in range(num_attacker)]
defenders_trajectory = [[] for _ in range(num_defender)]

# for plotting
attackers_x = [[] for _ in range(num_attacker)]
attackers_y = [[] for _ in range(num_attacker)]
defenders_x = [[] for _ in range(num_defender)]
defenders_y = [[] for _ in range(num_defender)]

# mip results 
capture_decisions = []

# load the initial states
current_attackers = attackers_initials
current_defenders = defenders_initials
for i in range(num_attacker):
    attackers_trajectory[i].append(current_attackers[i])
    attackers_x[i].append(current_attackers[i][0])
    attackers_y[i].append(current_attackers[i][1])
for j in range(num_defender):
    defenders_trajectory[j].append(current_defenders[j])
    defenders_x[j].append(current_defenders[j][0])
    defenders_y[j].append(current_defenders[j][1])

# initialize the captured results
attackers_status_logs = []
attackers_status = [0 for _ in range(num_attacker)]
stops_index = []  # the list stores the indexes of attackers that has been captured or arrived
attackers_status_logs.append(deepcopy(attackers_status))

# log the attackers be assigned defenders
attacker_assigneds = []
RA1v1s = []
RA1v2s = []
controls_defender0 = []
controls_defender1 = []

print("The simulation starts: \n")
# simulation starts
for _ in range(0, times):

    # MIP Optimization
    RA1v1 = capture_1vs1(current_attackers, current_defenders, v1v1, stops_index)  # attacker will win the 1 vs. 1 game
    RA2v1, value_list = capture_2vs1(current_attackers, current_defenders, v2v1)  # defender can not capture both attackers in 2 vs. 1 game
    RA1v2, RA1v2_ = capture_1vs2(current_attackers, current_defenders, v1v2)  # attacker will win the 1 vs. 2 game
    
    RA1v1s.append(RA1v1)
    RA1v2s.append(RA1v2)
    # print(f"The current step {_} RA1v2C is {RA1v2C}.")
    # Hanyang: first big change
    # selected = mip_solver(num_attacker, num_defender, RA2v1, RA1v1)
    # selected, weights, assigned = extend_mip_solver(num_attacker, num_defender, RA1v1, RA1v2, RA2v1)
    selected, weights, assigned = extend_mip_solver1(num_attacker, num_defender, RA1v1, RA1v2, RA1v2_, RA2v1)
    # print(f"In current step {_} the shape of weights is {weights.shape}.")
    # print(f"In current step {_} the assigned from attackers' views is {assigned}.")
    attacker_assigneds.append(assigned)
    # print(f"The current step {_} assignment weights is {weights}. \n")
    # print(f"The current step {_} e is {e}")
    
    # selected = mip_solver(num_attacker, num_defender, RA2v1, RA1v1)
    capture_decisions.append(selected)  # document the capture results

    # calculate the current controls of defenders
    control_defenders = [[] for num in range(num_defender)]  # current controls of defenders, [(d1xc, d1yc), (d2xc, d2yc)]
    calculated_defenders = []  # store the calculated defenders which should not calculate the control again
    for j in range(num_defender):
        
        # check whether the control of defender j has been calculated by 1 vs. 2 game
        if j in calculated_defenders:
            continue

        d1x, d1y = current_defenders[j]
        if len(selected[j]) == 2:  # defender j capture the attacker selected[j][0] and selected[j][1]
            a1x, a1y = current_attackers[selected[j][0]]
            a2x, a2y = current_attackers[selected[j][1]]
            joint_states2v1 = (a1x, a1y, a2x, a2y, d1x, d1y)
            control_defenders[j].append(defender_control2v1_1slice(agents_2v1, grid2v1, value2v1, tau2v1, joint_states2v1))

        elif len(selected[j]) == 1: # defender j capture the attacker selected[j][0]
            a1x, a1y = current_attackers[selected[j][0]]

            if weights[selected[j], j] == 0.5:  # use 1 vs. 2 game based control
                print(f"Use 1vs2 control in the current step {_}. \n")
                collaborate_defender = assigned[selected[j][0]][-1]  # the index of another defender
                d2x, d2y = current_defenders[collaborate_defender]
                joint_state1v2 = (a1x, a1y, d1x, d1y, d2x, d2y)
                opt_d1, opt_d2, opt_d3, opt_d4 = defender_control1vs2_slice(agents_1v2, grid1v2, value1v2, tau1v2, joint_state1v2)
                control_defenders[j].append((opt_d1, opt_d2))
                control_defenders[collaborate_defender].append((opt_d3, opt_d4))
                calculated_defenders.append(collaborate_defender)
                controls_defender0.append((opt_d1, opt_d2))  
                controls_defender1.append((opt_d3, opt_d4))
                
            else:  # use 1 vs. 1 game based control
                joint_states1v1 = (a1x, a1y, d1x, d1y)
                control_defenders[j].append(defender_control1v1_1slice(agents_1v1, grid1v1, value1v1, tau1v1, joint_states1v1))

        else:  # defender j could not capture any of attackers
            attacker_index = select_attacker2(d1x, d1y, current_attackers, stops_index)  # choose the nearest attacker
            a1x, a1y = current_attackers[attacker_index]
            joint_states1v1 = (a1x, a1y, d1x, d1y) 
            control_defenders[j].append((0.0, 0.0))  # defender_control1v1_1slice(agents_1v1, grid1v1, value1v1, tau1v1, joint_states1v1)

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
        attackers_x[i].append(current_attackers[i][0])
        attackers_y[i].append(current_attackers[i][1])

    for j in range(num_defender):
        defenders_trajectory[j].append(current_defenders[j])
        defenders_x[j].append(current_defenders[j][0])
        defenders_y[j].append(current_defenders[j][1])

    # check the attackers status: captured or not  
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

# print(f"The log of attackers assigned is {attacker_assigneds}. \n")
print(f"The MIP assignment result is {capture_decisions}. \n")
# print(f"The log of RA1v2 is {RA1v2s}. \n")
# print(f"The log of RA1v1 is {RA1v1s}. \n")
# Play the animation
animation_2v1(attackers_trajectory, defenders_trajectory, attackers_status_logs, T)

# print(f"The controls of defender0 is {controls_defender0}. \n")
# print(f"The controls of defender1 is {controls_defender1}. \n")

# # plot the trajectories seperately T = [0.475s (95 A1 by D0), 0.69s (138 A0 by D0)]
# if T == 0.475:
#     plot_simulation2v1_1(attackers_x, attackers_y, defenders_x, defenders_y)  #
# elif T == 0.69: ##  - slice 95
#     plot_simulation2v1_2(attackers_x, attackers_y, defenders_x, defenders_y)  #
# elif T == 0.785:
#     plot_simulation2v1_2s(attackers_x, attackers_y, defenders_x, defenders_y)  #
# else:
#     plot_simulation(attackers_x, attackers_y, defenders_x, defenders_y)
    # plot_simulation2v1_2(attackers_x, attackers_y, defenders_x, defenders_y)  


# # check the smallest distance
# # D and A1
# distances_d0a1 = []
# # print(f"The shape of defenders_traj is {len(defenders_trajectory[0])}")
# # print(f"The shape of attacker1_traj is {len(attackers_trajectory[1])}")

# for po in range(len(attackers_trajectory[1])):
#     a1 = attackers_trajectory[1][po]
#     d0 = defenders_trajectory[0][po]
#     print(f"A1 is at {a1}")
#     print(f"D0 is at {d0}")
#     distance = np.sqrt((a1[0] - d0[0])**2 + (a1[1] - d0[1])**2)
#     distances_d0a1.append(distance)

# minimum_distance = np.min(distances_d0a1)
# print(f"The smallest distance between D0 and A1 is {minimum_distance}.\n")
