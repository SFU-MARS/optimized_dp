from mip import *
from odp.Grid import Grid
from odp.solver import HJSolver, computeSpatDerivArray
from MRAG.AttackerDefender1v1 import AttackerDefender1v1 
from MRAG.AttackerDefender2v1 import AttackerDefender2v1
from utilities import *
import plotly.graph_objects as go
from plotly.graph_objects import Layout
import numpy as np

# information of the reach-avoid game
# num_attacker = 3
# num_defender = 2

# # establish P, Pas and Pc
# # pay attention to the order of initialization
# P, Pc, Pas = [], [], []
# # initialize 
# for j in range(num_defender):
#     P.append([])
# # print(P)
# # add constraints, this step is replaced by checking value function in the future
# P[0].append((0, 1))
# P[0].append((0, 2))
# P[1].append((0, 2))
# # print(P)
# for i in range(num_attacker):
#     for j in range(i+1, num_attacker):
#         Pas.append((i, j))
# # print(Pas)
# for j in range(num_defender):
#     Pc.append(list(set(Pas).difference(set(P[j]))))
# # print(Pc)

# # establish I, Ias and Ic
# I, Ias, Ic = [], [], []
# for j in range(num_defender):
#     I.append([])
# # add constraints, this step is replaced by checking value function in the future
# I[0].append(0)
# I[0].append(1)
# I[0].append(2)
# I[1].append(2)
# for i in range(num_attacker):
#     Ias.append(i)
# # print(Ias)  
# for j in range(num_defender):
#     Ic.append(list(set(Ias).difference(set(I[j]))))
# # print(Ic)


# # establish the MIP problem
# model = Model(solver_name=CBC) # use GRB for Gurobi, CBC default
# e = [[model.add_var(var_type=BINARY) for j in range(num_defender)] for i in range(num_attacker)] # e[attacker index][defender index]

# # add pair constraints
# # add constraints 12c
# for j in range(num_defender):
#     model += xsum(e[i][j] for i in range(num_attacker)) <= 2

# # add constraints 12d
# for i in range(num_attacker):
#     model += xsum(e[i][j] for j in range(num_defender)) <= 1

# # add constraints 12c
# for j in range(num_defender):
#     for pairs in (Pc[j]):
#         # print(pairs)
#         model += e[pairs[0]][j] + e[pairs[1]][j] <= 1

# # add constraints 12f
# for j in range(num_defender):
#     for indiv in (Ic[j]):
#         # print(indiv)
#         model += e[indiv][j] == 0

# # objective functions
# model.objective = maximize(xsum(e[i][j] for j in range(num_defender) for i in range(num_attacker)))

# # mip solve
# model.max_gap = 0.05
# status = model.optimize(max_seconds=300)
# if status == OptimizationStatus.OPTIMAL:
#     print('optimal solution cost {} found'.format(model.objective_value))
# elif status == OptimizationStatus.FEASIBLE:
#     print('sol.cost {} found, best possible: {} '.format(model.objective_value, model.objective_bound))
# elif status == OptimizationStatus.NO_SOLUTION_FOUND:
#     print('no feasible solution found, lower bound is: {} '.format(model.objective_bound))
# if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
#     print('Solution:')
#     selected = []
#     for j in range(num_defender):
#             selected.append([])
#             for i in range(num_attacker):
#                 if e[i][j].x >= 0.9:
#                     selected[j].append((i, j))
#     print(selected)

# current_attackers = [[(0.0, 0.0)], [(0.0, 0.8)], [(-0.5, 0.0)], [(0.5, -0.5)]]
# print(current_attackers[1][0])
# print(len(current_attackers))


value1v1 = np.load('MRAG/1v1AttackDefend.npy')
value2v1 = np.load('MRAG/2v1AttackDefend.npy')
grid1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45])) # original 45
grid2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([30, 30, 30, 30, 30, 30]))

agents_1v1 = AttackerDefender1v1(uMode="min", dMode="max")  # 1v1 (4 dims dynamics)
agents_2v1 = AttackerDefender2v1(uMode="min", dMode="max")  # 2v1 (6 dim dynamics)


# # Compute spatial derivatives at every state
# a1x_derivative = computeSpatDerivArray(grid1v1, value1v1, deriv_dim=1, accuracy="low")
# a1y_derivative = computeSpatDerivArray(grid1v1, value1v1, deriv_dim=2, accuracy="low")
# d1x_derivative = computeSpatDerivArray(grid1v1, value1v1, deriv_dim=3, accuracy="low")
# d2y_derivative = computeSpatDerivArray(grid1v1, value1v1, deriv_dim=4, accuracy="low")

# spat_deriv_vector = (a1x_derivative[10,20,15,15], a1y_derivative[10,20,15,15],
#                      d1x_derivative[10,20,15,15], d2y_derivative[10,20,15,15])


# # Compute spatial derivatives at every state
# a1x_derivative = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=1, accuracy="low")
# a1y_derivative = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=2, accuracy="low")
# a2x_derivative = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=3, accuracy="low")
# a2y_derivative = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=4, accuracy="low")
# d1x_derivative = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=5, accuracy="low")
# d2y_derivative = computeSpatDerivArray(grid2v1, value2v1, deriv_dim=6, accuracy="low")

# # Let's compute optimal control at some random idices
# spat_deriv_vector = (a1x_derivative[10,20,15,15,15,15], a1y_derivative[10,20,15,15,15,15],
#                      a2x_derivative[10,20,15,15,15,15], a2y_derivative[10,20,15,15,15,15],
#                      d1x_derivative[10,20,15,15,15,15], d2y_derivative[10,20,15,15,15,15])

# # Compute the optimal control
# opt_d1, opt_d2 = agents_1v1.optDstb_inPython(spat_deriv_vector)
# print("Optimal accel is {}\n".format(opt_d1))
# print("Optimal rotation is {}\n".format(opt_d2))

# # initialize positions of attackers and defenders
# attackers_initials = [(0.0, 0.0), (3.0, 0.0), (-5.0, 0.0), (6.0, 0.0)]
# defenders_initials = [(0.3, 0.5), (-0.3, 0.5)]
# num_attacker = len(attackers_initials)
# num_defender = len(defenders_initials)
# attackers_trajectory  = [[] for _ in range(num_attacker)]
# defenders_trajectory = [[] for _ in range(num_defender)]
# capture_decisions = []

# # simulation begins
# current_attackers = attackers_initials
# current_defenders = defenders_initials

# d1x = 1.0
# d1y = 0.0

# index = select_attacker(d1x, d1y, current_attackers)
# print(index)

# T = 0.6 # total simulation time
# deltat = 0.01 # calculation time interval
# times = int(T/deltat)


# plot_simulation()

# initialize positions of attackers and defenders
attackers_initials = [(0.0, 0.0), (0.0, 0.8), (-0.5, 0.0), (0.5, -0.5)]
defenders_initials = [(0.3, 0.5), (-0.3, 0.5)]
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

print(attackers_x)
# document the initial positions of attackers and defenders
for i in range(num_attacker):
    attackers_trajectory[i].append(current_attackers[i])
    attackers_x[i].append(current_attackers[i][0])
    attackers_y[i].append(current_attackers[i][1])

for j in range(num_defender):
    defenders_trajectory[j].append(current_defenders[j])
    defenders_x[j].append(current_defenders[j][0])
    defenders_y[j].append(current_defenders[j][1])

print(attackers_x)
