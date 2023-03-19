import numpy as np
import math
from mip import *
from odp.solver import computeSpatDerivArray
from MRAG.AttackerDefender1v0 import AttackerDefender1v0


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
    """ Returns a binary value, 1 means the defender could capture two attackers

    Args:
        value2v1 (ndarray): 2v1 HJ value function
        joint_states2v1 (tuple): state of (a1x, a1y, a2x, a2y, d1x, d1y)
    """
    a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice = lo2slice2v1(joint_states2v1)
    flag = value2v1[a1x_slice, a1y_slice, a2x_slice, a2y_slice, d1x_slice, d1y_slice]
    if flag > 0:
        return 1  # d1 could capture (a1, a2)
    else:
        return 0

# generate the capture pair list P and the capture pair complement list Pc
def capture_pair(attackers, defenders, value2v1):
    """ Returns a list Pc that contains all pairs of attackers that the defender couldn't capture, [[(a1, a2), (a2, a3)], ...]

    Args:
        attackers (list): positions (set) of all attackers, [(a1x, a1y), ...]
        defenders (list): positions (set) of all defenders, [(d1x, d1y), ...]
        value2v1 (ndarray): 2v1 HJ value function
    """
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
                    selected[j].append(i)
        print(selected)
    return selected

def spatial_derivatives(grids, value_function, accuracy="low"):
    """ Returns a tuple of derivatives that contain derivatives of all dimensions

    Args:
        grids (class): the initial set up of the HJ problem
        value_function (ndarray): the calculated HJ value function
        accuracy (string): the calculation accuracy, default "low"
    """
    dim = len(grids.grid_points)
    derivatives = []
    for i in range(1, dim+1):
        derivatives.append(computeSpatDerivArray(grids, value_function, deriv_dim=i, accuracy=accuracy))
    return tuple(derivatives)

def add_trajectory(trajectories, next_positions):
    """Return a updated trajectories (list) that contain trajectories of agents (attackers or defenders)

    Args: 
        trajectories (list): [[(a1x1, a1y1), ...], ...]
        next_positions (list): [(a1xi, a1yi), ...]
    """
    pass

def next_positions(current_positions, controls):
    """Return the next positions (list) of attackers or defenders

    Arg:
    current_positions (list): [(), (),...]
    controls (list): [(), (),...]
    """
    temp = []
    num = len(controls)
    for i in range(num):
        temp.append((current_positions[i][0]+controls[i][0], current_positions[i][1]+controls[i][1]))
    return temp

def attackers_control(grids, value_function, current_positions):
    """Return a list of 2-dimensional control inputs of all attackers based on the value function

    Args:
    grids (class): the corresponding Grid instance
    value_function (ndarray): 1v0 or 1v1 or 2v1 HJ reachability value function
    current_positions (list): the attacker(s), [(), (),...]
    """
    control_attackers = []
    agents_1v0 = AttackerDefender1v0(uMode="min", dMode="max")
    x1_derivative = computeSpatDerivArray(grids, value_function, deriv_dim=1, accuracy='low')
    x2_derivative = computeSpatDerivArray(grids, value_function, deriv_dim=2, accuracy='low')
    for position in current_positions:
        x1, x2 = lo2slice1v0(position)
        spat_deriv_vector = (x1_derivative[x1][x2], x2_derivative[x1][x2])
        control_attackers.append(agents_1v0.optCtrl_inPython(spat_deriv_vector))
    return control_attackers

def compute_control(grids, value_function, current_positions):
    pass