from mip import *
from sys import stdout as out

# information of the reach-avoid game
num_attacker = 3
num_defender = 2

# establish P, Pas and Pc
# pay attention to the order of initialization
P, Pc, Pas = [], [], []
# initialize 
for j in range(num_defender):
    P.append([])
# print(P)
# add constraints, this step is replaced by checking value function in the future
P[0].append((0, 1))
P[0].append((0, 2))
P[1].append((0, 2))
# print(P)
for i in range(num_attacker):
    for j in range(i+1, num_attacker):
        Pas.append((i, j))
# print(Pas)
Pc = []
for j in range(num_defender):
    Pc.append(list(set(Pas).difference(set(P[j]))))
# print(Pc)

# establish I, Ias and Ic
I, Ias, Ic = [], [], []
for j in range(num_defender):
    I.append([])
# add constraints, this step is replaced by checking value function in the future
I[0].append(0)
I[0].append(1)
I[0].append(2)
I[1].append(2)
for i in range(num_attacker):
    Ias.append(i)
# print(Ias)  
for j in range(num_defender):
    Ic.append(list(set(Ias).difference(set(I[j]))))
# print(Ic)


# establish the MIP problem
model = Model(solver_name=CBC) # use GRB for Gurobi, CBC default
e = [[model.add_var(var_type=BINARY) for j in range(num_defender)] for i in range(num_attacker)] # e[attacker index][defender index]

# add pair constraints
# add constraints 12c
for j in range(num_defender):
    model += xsum(e[i][j] for i in range(num_attacker)) <= 2

# add constraints 12d
for i in range(num_attacker):
    model += xsum(e[i][j] for j in range(num_defender)) <= 1

# add constraints 12c
for j in range(num_defender):
    for pairs in (Pc[j]):
        # print(pairs)
        model += e[pairs[0]][j] + e[pairs[1]][j] <= 1

# add constraints 12f
for j in range(num_defender):
    for indiv in (Ic[j]):
        # print(indiv)
        model += e[indiv][j] == 0

# objective functions
model.objective = maximize(xsum(e[i][j] for j in range(num_defender) for i in range(num_attacker)))

# mip solve
# model.optimize()

# check results
# selected = []
# for i in range(num_attacker):
#     for j in range(num_defender):
#         if e[i][j].x >= 0.5:
#             selected.append((i, j))
# print(selected)


model.max_gap = 0.05

status = model.optimize(max_seconds=300)
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(model.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {} '.format(model.objective_value, model.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {} '.format(model.objective_bound))
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    print('Solution:')
    selected = []
    for i in range(num_attacker):
        for j in range(num_defender):
            if e[i][j].x >= 0.9:
                selected.append((i, j))
    print(selected)