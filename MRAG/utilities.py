import numpy as np
from mip import *

# locations 2 slices
def lo2slice1v1(x_location, y_location, slices=45):
    x_slice = np.round((1 + x_location) * (slices - 1) / 2)
    y_slice = np.round((1 + y_location) * (slices - 1) / 2)
    return int(x_slice), int(y_slice)


# calculate the value function of the current state
def state_value(V, x1, y1, x2, y2, slices=45):
    # (x1, y1) and (x2, y2) are locations
    x1_slice, y1_slice = lo2slice1v1(x1, y1, slices)
    x2_slice, y2_slice = lo2slice1v1(x2, y2, slices)
    value = V[x1_slice, y1_slice, x2_slice, y2_slice, 0]  # 0 means the final tube
    return value


# check in the current state, the attacker is captured by the defender or not
def check1v1(value1v1, joint_states1v1):
    # inputs:
    # value1v1: the calculated HJ value function of the 1v1 game
    # joint_states1v1: a set contains (a1, d1)
    a1x, a1y, d1x, d1y = joint_states1v1
    flag = state_value(value1v1, a1x, a1y, d1x, d1y)
    if flag > 0:
        return 1 # d1 could capture a1
    else:
        return 0

# localizations to silces in 2v1 game
def lo2slice2v1(joint_states2v1, slices=30):
    # the input of the joint_states2v1 should be a set (a1x, a1y, a2x, a2y, d1x, d1y)
    a1x, a1y, a2x, a2y, d1x, d1y = joint_states2v1
    a1x_slice = int(np.round((1 + a1x) * (slices - 1) / 2)) 
    a1y_slice = int(np.round((1 + a1y) * (slices - 1) / 2)) 
    a2x_slice = int(np.round((1 + a2x) * (slices - 1) / 2))
    a2y_slice = int(np.round((1 + a2y) * (slices - 1) / 2))
    d1x_slice = int(np.round((1 + d1x) * (slices - 1) / 2))
    d1y_slice = int(np.round((1 + d1y) * (slices - 1) / 2))
    return a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice

# check the capture relationship in 2v1 game
def check2v1(value2v1, joint_states2v1):
    # inputs:
    # value2v1: the calculated HJ value function of the 2v1 game
    # joint_states2v1: a set contains all locations within the range of the game
    a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice = lo2slice2v1(joint_states2v1)
    flag = value2v1[a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice]
    if flag > 0:
        return 1  # d1 could capture (a1, a2)
    else:
        return 0

# generate the capture pair list P and the capture pair complement list Pc
def capture_pair(attackers, defenders, value2v1):
    # attackers is a list which contains positions of all attackers in the form of set: [(a1x, a1y),... (aMx, aMy)]
    # defenders is a list which contains positions of all defenders in the form of set: [(d1x, d1y),... (dNx, dNy)]
    # return is the capture pairs list P = [[(ai, ak)], ..., [()]]
    num_attacker, num_defender = len(attackers), len(defenders)
    P, Pc, Pas = [], [], []
    # generate P
    for j in range(num_defender):
        P.append([])
        djx, djy = defenders[j]
        for i in range(num_attacker):
            for k in range(i+1, num_attacker):
                aix, aiy = attackers[i]
                akx, aky = attackers[k]
                joint_states = (aix, aiy, akx, aky, djx, djy)
                if check2v1(value2v1, joint_states):
                    P[j].append((i, k))
    # generata Pas
    for i in range(num_attacker):
        for j in range(i+1, num_attacker):
            Pas.append((i, j))
    # generate Pc
    for j in range(num_defender):
        Pc.append(list(set(Pas).difference(set(P[j]))))
    return Pc

# generate the capture individual list I and the capture individual complement list Ic
def capture_individual(attackers, defenders, value1v1):
    # attackers is a list which contains positions of all attackers in the form of set: [(a1x, a1y),... (aMx, aMy)]
    # defenders is a list which contains positions of all defenders in the form of set: [(d1x, d1y),... (dNx, dNy)]
    # return is the capture individuals list I = [[a1, ai], ..., []]
    num_attacker, num_defender = len(attackers), len(defenders)
    I, Ic, Ias = [], [], []
    # generate I
    for j in range(num_defender):
        I.append([])
        djx, djy = defenders[j]
        for i in range(num_attacker):
            aix, aiy = attackers[i]
            joint_states = (aix, aiy, djx, djy)
            if check1v1(value1v1, joint_states):
                I[j].append(i)
    # generate Ias
    for i in range(num_attacker):
        Ias.append(i)
    # generate Ic
    for j in range(num_defender):
        Ic.append(list(set(Ias).difference(set(I[j]))))
    return Ic

# set up and solve the mixed integer programming question
def mip_solver(num_attacker, num_defender, Pc, Ic):
    # initialize the solver
    model = Model(solver_name=CBC) # use GRB for Gurobi, CBC default
    e = [[model.add_var(var_type=BINARY) for j in range(num_defender)] for i in range(num_attacker)] # e[attacker index][defender index]
    # add constraints
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
    # set up objective functions
    model.objective = maximize(xsum(e[i][j] for j in range(num_defender) for i in range(num_attacker)))
    # problem solving
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
        for j in range(num_defender):
            selected.append([])
            for i in range(num_attacker):
                if e[i][j].x >= 0.9:
                    selected[j].append((i, j))
        print(selected)
    return selected
        # todo: how to tell the capture is used by 2v1 or 1v1?
