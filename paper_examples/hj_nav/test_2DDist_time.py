import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar4D
# Plot options
from odp.Plots import PlotOptions, plot_isosurface, plot_valuefunction

# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import datetime

from pyinstrument import Profiler

def goal_dist(states: np.ndarray, goal: np.ndarray):
    # goal should be in the shape of (2,)
    if goal is None:
        return np.inf
    
    if states.ndim == 1:
        return np.linalg.norm(states - goal)
    else:
        return np.linalg.norm(states - goal, axis=1)


def dist_heuristic(states: np.ndarray, goal: np.ndarray):
    if states.ndim == 1:
        return goal_dist(states[:2], goal)
    else:
        goal_dist(states[:, :2], goal)

# Dynamics bounds: [x, y, v, theta]
minbounds = np.array([-8.0, -6.0, -0.2, 0.0])
maxbounds = np.array([8.0, 10.0, 0.8, 2*np.pi])
num_grids = np.array([200, 200, 100, 100])

goal = np.array([0.0, 0.0])

# # TTR loading
# ttr4d = np.load("paper_examples/hj_nav/data/ttr_a099bf1e7dd4fd57c4b657eb46a00bc92f7add253cba35a1c64a5c0988e03afd.npy")
# print(ttr4d.shape)
neighbours = np.random.uniform(low=minbounds, high=maxbounds, size=(35, 4))  # neighbour nodes (35, 4, 8) 35 neighbour nodes, 4 trajectory points, 8 dimensions

# Number of evaluation
num_times = 1e4

suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M')

# Evaluation
profiler = Profiler()
profiler.start()

for i in range(int(num_times)):
    dist_heuristic(neighbours, goal)

profiler.stop()
profiler.print()
# print(profiler.output_text(unicode=True, color=True))
# Save as HTML
with open(f'paper_examples/hj_nav/results/dist_{num_times}_evaluations_{suffix}.html', 'w') as f:
    f.write(profiler.output_html())
