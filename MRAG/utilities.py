import numpy as np
import math
from mip import *

# locations 2 slices
# def lo2slice1v1(x_location, y_location, slices=45):
#     x_slice = np.round((1 + x_location) * (slices - 1) / 2)
#     y_slice = np.round((1 + y_location) * (slices - 1) / 2)
#     return int(x_slice), int(y_slice)

def lo2slice1v1(joint_states1v1, slices=45):
    """ Returns a tuple of the closest index of each state in the grid

    Args:
        joint_states2v1 (tuple): state of (a1x, a1y, a2x, a2y, d1x, d1y)
        slices (int): number of grids, default 30
    """
    index = []
    grid_points = np.linspace(-1, +1, num=slices)
    for i, s in enumerate(joint_states1v1):
        idx = np.searchsorted(grid_points, s)
        if idx > 0 and (
            idx == len(grid_points)
            or math.fabs(s - grid_points[idx - 1])
            < math.fabs(s - grid_points[idx])
        ):
            index.append(idx - 1)
        else:
            index.append(idx)
    return tuple(index)

# check in the current state, the attacker is captured by the defender or not
def check1v1(value1v1, joint_states1v1):
    # inputs:
    # value1v1: the calculated HJ value function of the 1v1 game
    # joint_states1v1: a tuple contains (a1, d1)
    a1x_slice, a1y_slice, d1x_slice, d1y_slice = lo2slice1v1(joint_states1v1, slices=45)
    flag = value1v1[a1x_slice, a1y_slice, d1x_slice, d1y_slice]
    if flag > 0:
        return 1  # d1 could capture (a1, a2)
    else:
        return 0

# localizations to silces in 2v1 game
def lo2slice2v1(joint_states2v1, slices=30):
    """ Returns a tuple of the closest index of each state in the grid

    Args:
        joint_states2v1 (tuple): state of (a1x, a1y, a2x, a2y, d1x, d1y)
        slices (int): number of grids, default 30
    """
    index = []
    grid_points = np.linspace(-1, +1, num=slices)
    for i, s in enumerate(joint_states2v1):
        idx = np.searchsorted(grid_points, s)
        if idx > 0 and (
            idx == len(grid_points)
            or math.fabs(s - grid_points[idx - 1])
            < math.fabs(s - grid_points[idx])
        ):
            index.append(idx - 1)
        else:
            index.append(idx)
    return tuple(index)

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
    Pc = []
    # generate Pc
    for j in range(num_defender):
        Pc.append([])
        djx, djy = defenders[j]
        for i in range(num_attacker):
            for k in range(i+1, num_attacker):
                aix, aiy = attackers[i]
                akx, aky = attackers[k]
                joint_states = (aix, aiy, akx, aky, djx, djy)
                if not check2v1(value2v1, joint_states):
                    Pc[j].append((i, k))
    return Pc

# generate the capture individual list I and the capture individual complement list Ic
def capture_individual(attackers, defenders, value1v1):
    # attackers is a list which contains positions of all attackers in the form of set: [(a1x, a1y),... (aMx, aMy)]
    # defenders is a list which contains positions of all defenders in the form of set: [(d1x, d1y),... (dNx, dNy)]
    # return is the capture individuals list I = [[a1, ai], ..., []]
    num_attacker, num_defender = len(attackers), len(defenders)
    Ic = []
    # generate I
    for j in range(num_defender):
        Ic.append([])
        djx, djy = defenders[j]
        for i in range(num_attacker):
            aix, aiy = attackers[i]
            joint_states = (aix, aiy, djx, djy)
            if not check1v1(value1v1, joint_states):
                Ic[j].append(i)
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