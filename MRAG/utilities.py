import numpy as np
import math
import datetime
from mip import *
from odp.solver import computeSpatDerivArray

# localizations to silces in 1v0 game
def lo2slice1v0(joint_states1v0, slices=45):
    """ Returns a tuple of the closest index of each state in the grid

    Args:
        joint_states1v0 (tuple): state of (a1x, a1y)
        slices (int): number of grids, default 45
    """
    index = []
    grid_points = np.linspace(-1, +1, num=slices)
    for i, s in enumerate(joint_states1v0):
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

def lo2slice1v1(joint_states1v1, slices=45):
    """ Returns a tuple of the closest index of each state in the grid

    Args:
        joint_states1v1 (tuple): state of (a1x, a1y, d1x, d1y)
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
    """ Returns a binary value, 1 means the defender could capture the attacker

    Args:
        value1v1 (ndarray): 1v1 HJ value function
        joint_states1v1 (tuple): state of (a1x, a1y, d1x, d1y)
    """
    a1x_slice, a1y_slice, d1x_slice, d1y_slice = lo2slice1v1(joint_states1v1, slices=30)
    flag = value1v1[a1x_slice, a1y_slice, d1x_slice, d1y_slice]
    if flag > 0:
        return 1  # d1 could capture (a1)
    else:  # d1 could not capture a1
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
    """ Returns a binary value, 1 means the defender could capture two attackers

    Args:
        value2v1 (ndarray): 2v1 HJ value function
        joint_states2v1 (tuple): state of (a1x, a1y, a2x, a2y, d1x, d1y)
    """
    a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice = lo2slice2v1(joint_states2v1)
    flag = value2v1[a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice]
    # print("2v1 value is {}".format(flag))
    if flag > 0:
        return 1, flag  # d1 could capture (a1, a2) simutaneously
    else:
        return 0, flag

def check1v2(value1v2, joint_states1v2):
    """ Returns a binary value, True means the attacker would be captured by two defenders

        Args:
            value1v2 (ndarray): 1v2 HJ value function
            joint_states1v2 (tuple): state of (ax, ay, d1x, d1y, d2x, d2y)
    """
    ax_slice, ay_slice, d1x_slice, d1y_slice, d2x_slice, d2y_slice = lo2slice2v1(joint_states1v2)
    flag = value1v2[ax_slice, ay_slice, d1x_slice, d1y_slice, d2x_slice, d2y_slice]

    if flag > 0:  # d1 and d2 could capture a
        return 1, flag
    else:  # d1 and d2 could not capture a
        return 0, flag

# generate the capture pair list P and the capture pair complement list Pc
def capture_2vs1(attackers, defenders, value2v1):
    """ Returns a list Pc that contains all pairs of attackers that the defender couldn't capture, [[(a1, a2), (a2, a3)], ...]

    Args:
        attackers (list): positions (set) of all attackers, [(a1x, a1y), ...]
        defenders (list): positions (set) of all defenders, [(d1x, d1y), ...]
        value2v1 (ndarray): 2v1 HJ value function [, , , , ,]
    """
    num_attacker, num_defender = len(attackers), len(defenders)
    Pc = []
    values = []
    # generate Pc
    for j in range(num_defender):
        Pc.append([])
        values.append([])
        djx, djy = defenders[j]
        for i in range(num_attacker):
            for k in range(i+1, num_attacker):
                aix, aiy = attackers[i]
                akx, aky = attackers[k]
                joint_states = (aix, aiy, akx, aky, djx, djy)
                flag, val = check2v1(value2v1, joint_states)
                if not flag:
                    Pc[j].append((i, k))
                values[j].append(val)
    return Pc, values

def capture_pair2(attackers, defenders, value2v1, stops):
    """ Returns a list Pc that contains all pairs of attackers that the defender couldn't capture, [[(a1, a2), (a2, a3)], ...]

    Args:
        attackers (list): positions (set) of all attackers, [(a1x, a1y), ...]
        defenders (list): positions (set) of all defenders, [(d1x, d1y), ...]
        value2v1 (ndarray): 2v1 HJ value function
        stops (list): the captured attackers index
    """
    num_attacker, num_defender = len(attackers), len(defenders)
    Pc = []
    # generate Pc
    for j in range(num_defender):
        Pc.append([])
        djx, djy = defenders[j]
        for i in range(num_attacker):
            if i in stops:
                for k in range(i+1, num_attacker):
                    Pc[j].append((i, k))
            else:
                for k in range(i+1, num_attacker):
                    if k in stops:
                        Pc[j].append((i, k))
                    else:
                        aix, aiy = attackers[i]
                        akx, aky = attackers[k]
                        joint_states = (aix, aiy, akx, aky, djx, djy)
                        if not check2v1(value2v1, joint_states):
                            Pc[j].append((i, k))
    return Pc



def capture_1vs2(attackers, defenders, value1v2):
    #TODO: not finished, should not use dictionary or it will overwrite the former results
    num_attacker, num_defender = len(attackers), len(defenders)
    RA1v2 = [[] for _ in range(num_defender)]
    RA1v2_ = []
    # RA1v2C = []
    # generate RA1v2
    for j in range(num_defender):
        djx, djy = defenders[j]
        for k in range(j+1, num_defender):
            dkx, dky = defenders[k]
            for i in range(num_attacker):
                aix, aiy = attackers[i]
                joint_states = (aix, aiy, djx, djy, dkx, dky)
                flag, val = check1v2(value1v2, joint_states)
                if not flag:  # attacker i will win the 1 vs. 2 game
                    RA1v2[j].append(i)
                    RA1v2[k].append(i)
                    RA1v2_.append((i, j, k))
                # else:
                #     RA1v2C.append({i: (j, k)})
                    # RA1v2C.append((i, j, k))

    return RA1v2, RA1v2_

# generate the capture individual list I and the capture individual complement list Ic
def capture_individual(attackers, defenders, value1v1):
    """ Returns a list Ic that contains all attackers that the defender couldn't capture, [[a1, a3], ...]

    Args:
        attackers (list): positions (set) of all attackers, [(a1x, a1y), ...]
        defenders (list): positions (set) of all defenders, [(d1x, d1y), ...]
        value2v1 (ndarray): 1v1 HJ value function
    """
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

def capture_1vs1(attackers, defenders, value1v1, stops):
    """ Returns a list Ic that contains all attackers that the defender couldn't capture, [[a1, a3], ...]

    Args:
        attackers (list): positions (set) of all attackers, [(a1x, a1y), ...]
        defenders (list): positions (set) of all defenders, [(d1x, d1y), ...]
        value1v1 (ndarray): 1v1 HJ value function
        stops (list): the captured attackers index
    """
    num_attacker, num_defender = len(attackers), len(defenders)
    Ic = []
    # generate I
    for j in range(num_defender):
        Ic.append([])
        djx, djy = defenders[j]
        for i in range(num_attacker):
            if i in stops:  # ignore captured attackers
                Ic[j].append(i)
            else:
                aix, aiy = attackers[i]
                joint_states = (aix, aiy, djx, djy)
                if not check1v1(value1v1, joint_states):  # defender j could not capture attacker i
                    Ic[j].append(i)
    return Ic

# set up and solve the mixed integer programming question
def mip_solver(num_attacker, num_defender, Pc, Ic):
    """ Returns a list selected that contains all allocated attackers that the defender could capture, [[a1, a3], ...]

    Args:
        num_attackers (int): the number of attackers
        num_defenders (int): the number of defenders
        Pc (list): constraint pairs of attackers of every defender
        Ic (list): constraint individual attacker of every defender
    """
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
    # add constraints 12c Pc
    for j in range(num_defender):
        for pairs in (Pc[j]):
            # print(pairs)
            model += e[pairs[0]][j] + e[pairs[1]][j] <= 1
    # add constraints 12f Ic
    for j in range(num_defender):
        for indiv in (Ic[j]):
            # print(indiv)
            model += e[indiv][j] == 0
    # set up objective functions
    model.objective = maximize(xsum(e[i][j] for j in range(num_defender) for i in range(num_attacker)))
    # problem solving
    model.max_gap = 0.05
    # log_status = []
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
                    selected[j].append(i)
        print(selected)
    return selected


def extend_mip_solver(num_attacker, num_defender, RA1v1, RA1v2, RA2v1):
    """ Returns a list selected that contains all allocated attackers that the defender could capture, [[a1, a3], ...]

    Args:
        num_attackers (int): the number of attackers
        num_defenders (int): the number of defenders
        RA1v1 (list): the single indexes of attackers that will win the 1 vs. 1 game for each defender
        RA1v2 (list): the single indexes of attackers that will win the 1 vs. 2 game for each defender
        RA2v1 (list): the pair indexes of attackers that will not be captured together in the 2 vs. 1 game for each defender
    """
    # initialize the solver
    model = Model(solver_name=CBC) # use GRB for Gurobi, CBC default
    e = [[model.add_var(var_type=BINARY) for j in range(num_defender)] for i in range(num_attacker)] # e[attacker index][defender index]
    # initialize the weakly defend edges set W and their weights for each defender
    Weakly = [[] for _ in range(num_defender)]
    weights = np.ones((num_attacker, num_defender))
    # add constraints
    # add constraint 1: upper bound for attackers to be captured based on the 2 vs. 1 game
    for j in range(num_defender):
        model += xsum(e[i][j] for i in range(num_attacker)) <= 2

    # add constraint 2: upper bound for defenders to be assgined based on the 1 vs. 2 game
    for i in range(num_attacker):
        model += xsum(e[i][j] for j in range(num_defender)) <= 2

    # add constraint 3: the attacker i could not be captured by the defender j in both 1 vs. 1 and 1 vs. 2 games
    for j in range(num_defender):
        for attacker in RA1v1[j]:
            if attacker in RA1v2[j]:  # the attacker could win the defender in both 1 vs. 1 and 1 vs. 2 games
                model += e[attacker][j] == 0
            else:  # the attacker could win the defender in 1 vs. 1 game but not in 1 vs. 2 game
                Weakly[j].append(attacker)
                weights[attacker][j] = 0.5

    # add constraint 4: upper bound for attackers to be captured based on the 2 vs. 1 game result
    for j in range(num_defender):
        for pairs in (RA2v1[j]):
            # print(pairs)
            model += e[pairs[0]][j] + e[pairs[1]][j] <= 1

    # add constraint 5: upper bound for weakly defended attackers
    for j in range(num_defender):
        for indiv in (Weakly[j]):
            # print(indiv)
            model += e[indiv][j] <= xsum(e[indiv][k] for k in range(num_defender))
            
    # set up objective functions
    model.objective = maximize(xsum(weights[i][j] * e[i][j] for j in range(num_defender) for i in range(num_attacker)))
    # problem solving
    model.max_gap = 0.05
    # log_status = []
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
        assigned = [[] for _ in range(num_attacker)]
        for j in range(num_defender):
            selected.append([])
            for i in range(num_attacker):
                if e[i][j].x >= 0.9:
                    selected[j].append(i)
                    assigned[i].append(j)
        # print(f"The selected results in the extend_mip_solver is {selected}.")

    return selected, weights, assigned

def extend_mip_solver1(num_attacker, num_defender, RA1v1, RA1v2, RA1v2_, RA2v1):
    """ Returns a list selected that contains all allocated attackers that the defender could capture, [[a1, a3], ...]

    Args:
        num_attackers (int): the number of attackers
        num_defenders (int): the number of defenders
        RA1v1 (list): the single indexes of attackers that will win the 1 vs. 1 game for each defender
        RA1v2 (list): the single indexes of attackers that will win the 1 vs. 2 game for each defender
        RA2v1 (list): the pair indexes of attackers that will not be captured together in the 2 vs. 1 game for each defender
    """
    # initialize the solver
    model = Model(solver_name=CBC) # use GRB for Gurobi, CBC default
    e = [[model.add_var(var_type=BINARY) for j in range(num_defender)] for i in range(num_attacker)] # e[attacker index][defender index]
    # initialize the weakly defend edges set W and their weights for each defender
    Weakly = [[] for _ in range(num_defender)]
    weights = np.ones((num_attacker, num_defender))
    # add constraints
    # add constraint 1: upper bound for attackers to be captured based on the 2 vs. 1 game
    for j in range(num_defender):
        model += xsum(e[i][j] for i in range(num_attacker)) <= 2

    # add constraint 2: upper bound for defenders to be assgined based on the 1 vs. 2 game
    for i in range(num_attacker):
        model += xsum(e[i][j] for j in range(num_defender)) <= 2
    for add in RA1v2_:
        model += xsum(e[add[0]][j] for j in range(num_defender)) <= 1
    #     model += e[add[0]][add[1]] + e[add[0]][add[2]] <= 1
    # for i in range(num_attacker):
    #     model += xsum(e[i][j] for j in range(num_defender)) <= 2

    # add constraint 3: the attacker i could not be captured by the defender j in both 1 vs. 1 and 1 vs. 2 games
    for j in range(num_defender):
        for attacker in RA1v1[j]:
            if attacker in RA1v2[j]:  # the attacker could win the defender in both 1 vs. 1 and 1 vs. 2 games
                model += e[attacker][j] == 0
            else:  # the attacker could win the defender in 1 vs. 1 game but not in 1 vs. 2 game
                Weakly[j].append(attacker)
                weights[attacker][j] = 0.5

    # add constraint 4: upper bound for attackers to be captured based on the 2 vs. 1 game result
    for j in range(num_defender):
        for pairs in (RA2v1[j]):
            # print(pairs)
            model += e[pairs[0]][j] + e[pairs[1]][j] <= 1

    # add constraint 5: upper bound for weakly defended attackers
    for j in range(num_defender):
        for indiv in (Weakly[j]):
            # print(indiv)
            model += e[indiv][j] <= xsum(e[indiv][k] for k in range(num_defender))
            
    # set up objective functions
    model.objective = maximize(xsum(weights[i][j] * e[i][j] for j in range(num_defender) for i in range(num_attacker)))
    # problem solving
    model.max_gap = 0.05
    # log_status = []
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
        assigned = [[] for _ in range(num_attacker)]
        for j in range(num_defender):
            selected.append([])
            for i in range(num_attacker):
                if e[i][j].x >= 0.9:
                    selected[j].append(i)
                    assigned[i].append(j)
        # print(f"The selected results in the extend_mip_solver is {selected}.")

    return selected, weights, assigned


def next_positions(current_positions, controls, tstep):
    """Return the next positions (list) of attackers or defenders

    Arg:
    current_positions (list): [(), (),...]
    controls (list): [(), (),...]
    """
    temp = []
    num = len(controls)
    for i in range(num):
        temp.append((current_positions[i][0]+controls[i][0]*tstep, current_positions[i][1]+controls[i][1]*tstep))
    return temp

def next_positions_d(current_positions, controls, tstep):
    """Return the next positions (list) of attackers or defenders

    Arg:
    current_positions (list): [(), (),...]
    controls (list): [[(a, b)], [()],...]
    """
    temp = []
    num = len(controls)
    for i in range(num):
        temp.append((current_positions[i][0]+controls[i][0][0]*tstep, current_positions[i][1]+controls[i][0][1]*tstep))
    return temp

def next_positions_a(current_positions, controls, tstep, captured):
    """Return the next positions (list) of attackers considering the current captured results

    Arg:
    current_positions (list): [(), (),...]
    controls (list): [(), (),...]
    tstep (float): time step
    captured (list): the captured attackers
    """
    temp = []
    num = len(controls)
    if len(captured) == 0:
        for i in range(num):
            temp.append((current_positions[i][0]+controls[i][0]*tstep, current_positions[i][1]+controls[i][1]*tstep))
    else:
        for i in range(num):
            if not captured[i]:  # the attacker i has not been captured
                temp.append((current_positions[i][0]+controls[i][0]*tstep, current_positions[i][1]+controls[i][1]*tstep))
            else:
                temp.append((current_positions[i][0], current_positions[i][1]))
    return temp

def next_positions_a2(current_positions, controls, tstep, stops):
    """Return the next positions (list) of attackers considering the current captured results

    Arg:
    current_positions (list): [(), (),...]
    controls (list): [(), (),...]
    tstep (float): time step
    stops (list): the captured attackers
    """
    temp = []
    num = len(controls)
    for i in range(num):
        if i in stops:
            temp.append((current_positions[i][0], current_positions[i][1]))
        else:
            temp.append((current_positions[i][0]+controls[i][0]*tstep, current_positions[i][1]+controls[i][1]*tstep))
    return temp

def distance(attacker, defender):
    """Return the 2-norm distance between the attacker and the defender

    Args:
    attacker (tuple): the position of the attacker
    defender (tuple): the position of the defender
    """
    d = np.sqrt((attacker[0]-defender[0])**2 + (attacker[1]-defender[1])**2)
    return d

def select_attacker(d1x, d1y, current_attackers):
    """Return the nearest attacker index

    Args:
    d1x (float): the x position of the current defender
    d1y (float): the y position of the current defender
    current_attackers (list): the positions of all attackers, [(), (),...]
    stops_index (list): contains the indexes of attackers that has been captured
    """
    num = len(current_attackers)
    index = 0
    d = distance(current_attackers[index], (d1x, d1y))
    for i in range(1, num):
        temp = distance(current_attackers[i], (d1x, d1y))
        if temp <= d:
            index = i
    return index

def select_attacker2(d1x, d1y, current_attackers, stops_index):
    """Return the nearest attacker index

    Args:
    d1x (float): the x position of the current defender
    d1y (float): the y position of the current defender
    current_attackers (list): the positions of all attackers, [(), (),...]
    stops_index (list): contains the indexes of attackers that has been captured
    """
    num = len(current_attackers)
    for index in range(num):
        if index not in stops_index:
            break
    d = distance(current_attackers[index], (d1x, d1y))
    for i in range(index, num):
        if i not in stops_index:
            temp = distance(current_attackers[i], (d1x, d1y))
            if temp <= d:
                index = i
    return index

def defender_control1v1_v0(agents_1v1, joint_states1v1, a1x_1v1, a1y_1v1, d1x_1v1, d1y_1v1):
    """Return a list of 2-dimensional control inputs of one defender based on the value function
    
    Args:
    grid1v1 (class): the corresponding Grid instance
    value1v1 (ndarray): 1v1 HJ reachability value function with final time slice 
    agents_1v1 (class): the corresponding AttackerDefender instance
    joint_states1v1 (tuple): the corresponding positions of (A1, D1)
    """
    a1x, a1y, d1x, d2y = lo2slice1v1(joint_states1v1)

    spat_deriv_vector = (a1x_1v1[a1x, a1y, d1x, d2y], a1y_1v1[a1x, a1y, d1x, d2y],
                     d1x_1v1[a1x, a1y, d1x, d2y], d1y_1v1[a1x, a1y, d1x, d2y])

    opt_d1, opt_d2 = agents_1v1.optDstb_inPython(spat_deriv_vector)
    return (opt_d1, opt_d2)

def defender_control2v1_v0(agents_2v1, joint_states2v1, a1x_2v1, a1y_2v1, a2x_2v1, a2y_2v1, d1x_2v1, d1y_2v1):
    """Return a list of 2-dimensional control inputs of one defender based on the value function
    
    Args:
    grid2v1 (class): the corresponding Grid instance
    value2v1 (ndarray): 2v1 HJ reachability value function with final time slice 
    agents_2v1 (class): the corresponding AttackerDefender instance
    joint_states2v1 (tuple): the corresponding positions of (A1, A2, D1)
    """
    a1x, a1y, a2x, a2y, d1x, d1y = lo2slice2v1(joint_states2v1)

    spat_deriv_vector = (a1x_2v1[a1x, a1y, a2x, a2y, d1x, d1y], a1y_2v1[a1x, a1y, a2x, a2y, d1x, d1y],
                     a2x_2v1[a1x, a1y, a2x, a2y, d1x, d1y], a2y_2v1[a1x, a1y, a2x, a2y, d1x, d1y],
                     d1x_2v1[a1x, a1y, a2x, a2y, d1x, d1y], d1y_2v1[a1x, a1y, a2x, a2y, d1x, d1y])
    
    opt_d1, opt_d2 = agents_2v1.optDstb_inPython(spat_deriv_vector)
    return (opt_d1, opt_d2)
    
def bi_graph(value1v1, current_attackers, current_defenders, stops_index):
    """
    Return a bipartite graph (list: num_attackers x num_defenders) that contains the relationship between every pair of attackers and defenders

    Args:   
        value1v1 (ndarray): not including all the time slices, shape = [grid.dims, ..., grid.dims]
        current_attackers (list): a list that contains all the current positions of all attackers
        current_defenders (list): a list that contains all the current positions of all defenders
    """
    num_attacker = len(current_attackers)
    num_defender = len(current_defenders)
    bigraph = [[] for _ in range(num_attacker)]
    # generate a bipartite graph with num_attacker lines and num_defender columns
    for i in range(num_attacker):
        a1x, a1y = current_attackers[i]
        if i in stops_index:
            for j in range(num_defender):
                d1x, d1y = current_defenders[j]
                jointstate1v1 = (a1x, a1y, d1x, d1y)
                bigraph[i].append(0)
        else:
            for j in range(num_defender):
                d1x, d1y = current_defenders[j]
                jointstate1v1 = (a1x, a1y, d1x, d1y)
                if check1v1(value1v1, jointstate1v1):  # the defender could capture the attacker
                    bigraph[i].append(1)
                else:
                    bigraph[i].append(0)
    return bigraph

def spa_deriv(index, V, g, periodic_dims=[]):
    """
    Calculates the spatial derivatives of V at an index for each dimension
    From Michael
    Args:
        index: (a1x, a1y)
        V (ndarray): [..., neg2pos] where neg2pos is a list [scalar] or []
        g (class): the instance of the corresponding Grid
        periodic_dims (list): the corrsponding periodical dimensions []

    Returns:
        List of left and right spatial derivatives for each dimension
    """
    spa_derivatives = []
    for dim, idx in enumerate(index):
        if dim == 0:
            left_index = []
        else:
            left_index = list(index[:dim])

        if dim == len(index) - 1:
            right_index = []
        else:
            right_index = list(index[dim + 1:])

        next_index = tuple(
            left_index + [index[dim] + 1] + right_index
        )
        prev_index = tuple(
            left_index + [index[dim] - 1] + right_index
        )

        if idx == 0:
            if dim in periodic_dims:
                left_periodic_boundary_index = tuple(
                    left_index + [V.shape[dim] - 1] + right_index
                )
                left_boundary = V[left_periodic_boundary_index]
            else:
                left_boundary = V[index] + np.abs(V[next_index] - V[index]) * np.sign(V[index])
            left_deriv = (V[index] - left_boundary) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]
        elif idx == V.shape[dim] - 1:
            if dim in periodic_dims:
                right_periodic_boundary_index = tuple(
                    left_index + [0] + right_index
                )
                right_boundary = V[right_periodic_boundary_index]
            else:
                right_boundary = V[index] + np.abs(V[index] - V[prev_index]) * np.sign([V[index]])
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (right_boundary - V[index]) / g.dx[dim]
        else:
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]

        spa_derivatives.append(((left_deriv + right_deriv) / 2)[0])
    return spa_derivatives

def find_sign_change1v0(grid1v0, value1v0, current_state):
    """Return two positions (neg2pos, pos2neg) of the value function

    Args:
    grid1v0 (class): the instance of grid
    value1v0 (ndarray): including all the time slices, shape = [100, 100, len(tau)]
    current_state (tuple): the current state of the attacker
    """
    current_slices = grid1v0.get_index(current_state)
    current_value = value1v0[current_slices[0], current_slices[1], :]  # current value in all time slices
    neg_values = (current_value<=0).astype(int)  # turn all negative values to 1, and all positive values to 0
    checklist = neg_values - np.append(neg_values[1:], neg_values[-1])
    # neg(True) - pos(False) = 1 --> neg to pos
    # pos(False) - neg(True) = -1 --> pos to neg
    return np.where(checklist==1)[0], np.where(checklist==-1)[0]


def find_sign_change1v1(grid1v1, value1v1, jointstate1v1):
    """Return two positions (neg2pos, pos2neg) of the value function

    Args:
    grid1v1 (class): the instance of grid
    value1v1 (ndarray): including all the time slices, shape = [45, 45, 45, 45, len(tau)]
    jointstate1v1 (tuple): the current joint state of (a1x, a1y, d1x, d1y)
    """
    current_slices = grid1v1.get_index(jointstate1v1)
    current_value = value1v1[current_slices[0], current_slices[1], current_slices[2], current_slices[3], :]  # current value in all time slices
    neg_values = (current_value<=0).astype(int)  # turn all negative values to 1, and all positive values to 0
    checklist = neg_values - np.append(neg_values[1:], neg_values[-1])
    # neg(True) - pos(False) = 1 --> neg to pos
    # pos(False) - neg(True) = -1 --> pos to neg
    return np.where(checklist==1)[0], np.where(checklist==-1)[0]

def find_sign_change2v1(grid2v1, value2v1, jointstate2v1):
    """Return two positions (neg2pos, pos2neg) of the value function

    Args:
    grid2v1 (class): the instance of grid
    value2v1 (ndarray): including all the time slices, shape = [45, 45, 45, 45, len(tau)]
    jointstate2v1 (tuple): the current joint state of (a1x, a1y, d1x, d1y)
    """
    current_slices = grid2v1.get_index(jointstate2v1)
    current_value = value2v1[current_slices[0], current_slices[1], current_slices[2], current_slices[3], 
                             current_slices[4], current_slices[5], :]  # current value in all time slices
    neg_values = (current_value<=0).astype(int)  # turn all negative values to 1, and all positive values to 0
    checklist = neg_values - np.append(neg_values[1:], neg_values[-1])
    # neg(True) - pos(False) = 1 --> neg to pos
    # pos(False) - neg(True) = -1 --> pos to neg
    return np.where(checklist==1)[0], np.where(checklist==-1)[0]

def compute_control1v0(agents_1v0, grid1v0, value1v0, tau1v0, position, neg2pos):
    """Return the optimal controls (tuple) of the attacker
    Notice: calculate the spatial derivative vector in the game
    Args:
    agents_1v0 (class): the instance of 1v0 attacker defender
    grid1v0 (class): the instance of grid
    value1v0 (ndarray): 1v0 HJ reachability value function with all time slices
    tau1v0 (ndarray): all time indices
    current_state (tuple): the current state of the attacker
    x1_1v0 (ndarray): spatial derivative array of the first dimension
    x2_1v0 (ndarray): spatial derivative array of the second dimension
    """
    assert value1v0.shape[-1] == len(tau1v0)  # check the shape of value function

    # check the current state is in the reach-avoid set
    current_value = grid1v0.get_value(value1v0[..., 0], list(position))
    if current_value > 0:
        value1v0 = value1v0 - current_value
    
    # calculate the derivatives
    v = value1v0[..., neg2pos] # Minh: v = value1v0[..., neg2pos[0]]
    # print(f"The shape of the input value function v of attacker is {v.shape}. \n")
    start_time = datetime.datetime.now()
    spat_deriv_vector = spa_deriv(grid1v0.get_index(position), v, grid1v0)
    end_time = datetime.datetime.now()
    # print(f"The calculation of 2D spatial derivative vector is {end_time-start_time}. \n")
    return agents_1v0.optCtrl_inPython(spat_deriv_vector)

def attackers_control(agents_1v0, grid1v0, value1v0, tau1v0, current_attackers):
    """Return a list of 2-dimensional control inputs of all attackers based on the value function
    Notice: calculate the spatial derivative vector in the game
    Args:
    agents_1v0 (class): the instance of 1v0 attacker defender
    grid1v0 (class): the corresponding Grid instance
    value1v0 (ndarray): 1v0 HJ reachability value function with all time slices
    tau1v0 (ndarray): all time indices
    agents_1v0 (class): the corresponding AttackerDefender instance
    current_positions (list): the attacker(s), [(), (),...]
    x1_1v0 (ndarray): spatial derivative array of the first dimension
    x2_1v0 (ndarray): spatial derivative array of the second dimension
    """
    control_attackers = []
    for position in current_attackers:
        neg2pos, pos2neg = find_sign_change1v0(grid1v0, value1v0, position)
        # print(f"The neg2pos is {neg2pos}.\n")
        if len(neg2pos):
            control_attackers.append(compute_control1v0(agents_1v0, grid1v0, value1v0, tau1v0, position, neg2pos))
        else:
            control_attackers.append((0.0, 0.0))
    return control_attackers

def compute_control1v1(agents_1v1, grid1v1, value1v1, tau1v1, jointstate1v1, neg2pos):
    """Return the optimal controls (tuple) of the defender in 1v1 reach-avoid game
    Notice: calculate the spatial derivative vector in the game
    Args:
    agents_1v1 (class): the instance of 1v1 attacker defender
    grid1v1 (class): the instance of grid
    value1v1 (ndarray): 1v1 HJ reachability value function with all time slices
    tau1v1 (ndarray): all time indices
    jointstate1v1 (tuple): the current joint state of the attacker and the defender
    """
    assert value1v1.shape[-1] == len(tau1v1)  # check the shape of value function

    # check the current state is in the reach-avoid set
    current_value = grid1v1.get_value(value1v1[..., 0], list(jointstate1v1))
    if current_value > 0:
        value1v1 = value1v1 - current_value
    
    # calculate the derivatives
    v = value1v1[..., neg2pos] # Minh: v = value1v0[..., neg2pos[0]]
    start_time = datetime.datetime.now()
    spat_deriv_vector = spa_deriv(grid1v1.get_index(jointstate1v1), v, grid1v1)
    end_time = datetime.datetime.now()
    print(f"The calculation of 2D spatial derivative vector is {end_time-start_time}. \n")
    return agents_1v1.optDstb_inPython(spat_deriv_vector)

def defender_control1v1(agents_1v1, grid1v1, value1v1, tau1v1, jointstate1v1):
    """Return a list of 2-dimensional control inputs of one defender based on the value function
    
    Args:
    grid1v1 (class): the corresponding Grid instance
    value1v1 (ndarray): 1v1 HJ reachability value function with all time slices
    agents_1v1 (class): the corresponding AttackerDefender instance
    joint_states1v1 (tuple): the corresponding positions of (A1, D1)
    """
    neg2pos, pos2neg = find_sign_change1v1(grid1v1, value1v1, jointstate1v1)
    if len(neg2pos):
        opt_d1, opt_d2 = compute_control1v1(agents_1v1, grid1v1, value1v1, tau1v1, jointstate1v1, neg2pos)
    else:
        opt_d1, opt_d2 = 0.0, 0.0
    return (opt_d1, opt_d2)

def compute_control2v1(agents_2v1, grid2v1, value2v1, tau2v1, jointstate2v1, neg2pos):
    """Return the optimal controls (tuple) of the defender in 1v1 reach-avoid game
    NOT FINISHED YET!!!!!!
    Notice: calculate the spatial derivative vector in the game 
    Args:
    agents_2v1 (class): the instance of 2v1 attacker defender
    grid2v1 (class): the instance of grid
    value2v1 (ndarray): 2v1 HJ reachability value function with only final time slice
    tau2v1 (ndarray): all time indices
    jointstate2v1 (tuple): the current joint state of the attacker and the defender
    """
    # assert value2v1.shape[-1] == len(tau2v1)  # check the shape of value function

    # check the current state is in the reach-avoid set
    current_value = grid2v1.get_value(value2v1[..., 0], list(jointstate2v1))
    if current_value > 0:
        value2v1 = value2v1 - current_value
    
    # calculate the derivatives
    v = value2v1[..., neg2pos] # Minh: v = value1v0[..., neg2pos[0]]
    start_time = datetime.datetime.now()
    spat_deriv_vector = spa_deriv(grid2v1.get_index(jointstate2v1), v, grid2v1)
    end_time = datetime.datetime.now()
    print(f"The calculation of 2D spatial derivative vector is {end_time-start_time}. \n")
    return agents_2v1.optDstb_inPython(spat_deriv_vector)

def defender_control2v1(agents_2v1, grid2v1, value2v1, tau2v1, jointstate2v1):
    """Return a list of 2-dimensional control inputs of one defender based on the value function
    NOT FINISHED YET !!!
    Args:
    grid2v1 (class): the corresponding Grid instance
    value2v1 (ndarray): 2v1 HJ reachability value function  
    agents_2v1 (class): the corresponding AttackerDefender instance
    joint_states2v1 (tuple): the corresponding positions of (A1, D1)
    """
    neg2pos, pos2neg = find_sign_change1v1(grid2v1, value2v1, jointstate2v1)
    if len(neg2pos):
        opt_d1, opt_d2 = compute_control1v1(agents_2v1, grid2v1, value2v1, tau2v1, jointstate2v1, neg2pos)
    else:
        opt_d1, opt_d2 = 0.0, 0.0
    return (opt_d1, opt_d2)


def defender_control1v1_1slice(agents_1v1, grid1v1, value1v1, tau1v1, jointstate1v1):
    """Return a list of 2-dimensional control inputs of one defender based on the value function
    
    Args:
    grid1v1 (class): the corresponding Grid instance
    value1v1 (ndarray): 1v1 HJ reachability value function with only final slice
    agents_1v1 (class): the corresponding AttackerDefender instance
    joint_states1v1 (tuple): the corresponding positions of (A1, D1)
    """
    # calculate the derivatives
    # v = value1v1[...] # Minh: v = value1v0[..., neg2pos[0]]
    start_time = datetime.datetime.now()
    # print(f"The shape of the input value1v1 of defender is {value1v1.shape}. \n")
    spat_deriv_vector = spa_deriv(grid1v1.get_index(jointstate1v1), value1v1, grid1v1)
    opt_d1, opt_d2 = agents_1v1.optDstb_inPython(spat_deriv_vector)
    end_time = datetime.datetime.now()
    # print(f"The calculation of 4D spatial derivative vector is {end_time-start_time}. \n")
    return (opt_d1, opt_d2)

def defender_control2v1_1slice(agents_2v1, grid2v1, value2v1, tau2v1, jointstate2v1):
    """Return a list of 2-dimensional control inputs of one defender based on the value function
    
    Args:
    grid2v1 (class): the corresponding Grid instance
    value2v1 (ndarray): 1v1 HJ reachability value function with only final slice
    agents_2v1 (class): the corresponding AttackerDefender instance
    joint_states2v1 (tuple): the corresponding positions of (A1, A2, D)
    """
    # calculate the derivatives
    start_time = datetime.datetime.now()
    # print(f"The shape of the input value2v1 of defender is {value2v1.shape}. \n")
    spat_deriv_vector = spa_deriv(grid2v1.get_index(jointstate2v1), value2v1, grid2v1)
    opt_d1, opt_d2 = agents_2v1.optDstb_inPython(spat_deriv_vector)
    end_time = datetime.datetime.now()
    # print(f"The calculation of 6D spatial derivative vector is {end_time-start_time}. \n")
    return (opt_d1, opt_d2)

def defender_control1vs2_slice(agents_1v2, grid1v2, value1v2, tau1v2, jointstate1v2):
    """Return a list of 2-dimensional control inputs of one defender based on the value function
    
    Args:
    grid1v2 (class): the corresponding Grid instance
    value1v2 (ndarray): 1v2 HJ reachability value function with only final slice
    agents_1v2 (class): the corresponding AttackerDefender instance
    joint_states1v2 (tuple): the corresponding positions of (A, D1, D2)
    """
    # calculate the derivatives
    start_time = datetime.datetime.now()
    # print(f"The shape of the input value1v2 of defender is {value1v2.shape}. \n")
    spat_deriv_vector = spa_deriv(grid1v2.get_index(jointstate1v2), value1v2, grid1v2)
    opt_d1, opt_d2, opt_d3, opt_d4  = agents_1v2.optDstb_inPython(spat_deriv_vector)
    end_time = datetime.datetime.now()
    # print(f"The calculation of 6D spatial derivative vector is {end_time-start_time}. \n")
    return (opt_d1, opt_d2, opt_d3, opt_d4)

def capture_check(current_attackers, current_defenders, selected, last_captured):
    """
    Return a list that contains 0 or 1, 1 means this attacker is captured

    Args:
    current_attackers (list): the current states of all attackers
    current_defenders (list): the current states of all defenders
    selected (list): the capture relationship
    last_captured (list): the captured result of last time step
    """
    captured = last_captured
    # check the attacker in selected is captured by the defender or not
    for j in range(len(current_defenders)):
        if len(selected[j]):
            for i in selected[j]:
                if distance(current_defenders[j], current_attackers[i]) <= 0.1:
                    captured[i] = 1
                    print(f"The attacker{i} has been captured by the defender{j}! \n")

    return captured

def capture_check1(current_attackers, current_defenders, selected):
    """
    Return a list that contains 0 or 1, 1 means this attacker is captured

    Args:
    current_attackers (list): the current states of all attackers
    current_defenders (list): the current states of all defenders
    selected (list): the capture relationship
    """
    captured_status = [0 for _ in range(len(current_attackers))]
    for j in range(len(current_defenders)):
        if len(selected[j]):
            for i in selected[j]:
                if distance(current_defenders[j], current_attackers[i]) <= 0.1:
                    captured_status[i] = 1
    return captured_status

def check_status(old_captured, new_captured):
    changed = 0  # 
    num_attacker = len(old_captured)
    for i in range(num_attacker):
        if old_captured[i] == new_captured[i]:
            continue
        else:
            changed = 1
    return changed

def stoped_check(attackers_status, attackers_arrived):
    index = []
    for i, capture in enumerate(attackers_status):
        if capture:
            index.append(i)
    for j, arrived in enumerate(attackers_arrived):
        if arrived:
            index.append(j)
    return sorted(index)

def arrived_check(current_attackers):
    num = len(current_attackers)
    arrived = [0 for _ in range(num)]
    # check the attacker has arrived at the target set or not
    for i in range(num):
        if (0.6<=current_attackers[i][0]) and (current_attackers[i][0]<=0.8):
            if (0.1<=current_attackers[i][1]) and (current_attackers[i][1]<=0.3):
                arrived[i] = 1
                print(f"The attacker{i} has arrived at the target set! \n")
    return arrived

def matching_check(selected, stops_index):
    # return the checked selected list and the actual number of maximum matching
    for j in range(len(selected)):
        if len(selected[j]): # not empty
            for attacker in selected[j]:
                if attacker in stops_index:
                    selected[j].remove(attacker)
    num = 0
    for j in range(len(selected)):
        num += len(selected[j])
    return selected, num