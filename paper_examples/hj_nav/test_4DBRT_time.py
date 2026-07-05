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

# Dynamics bounds: [x, y, v, theta]
minbounds = np.array([-6.5, -13.0, -0.2, 0.0])
maxbounds = np.array([5.05, 15.05, 0.8, 2*np.pi])
num_grids = np.array([231, 150, 50, 50])

# Grid 
grid = Grid(minBounds=minbounds,
            maxBounds=maxbounds,
            dims=4,
            pts_each_dim=num_grids,
            periodicDims=[3])

# BRT loading
brt4d = np.load("paper_examples/hj_nav/data/BRT_Avoid_645aaa9aba6b5197871848d6c4409097b8af5d01aee611874ef2d57f685157bb.npy")
print(brt4d.shape)
neighbours = np.random.uniform(low=minbounds, high=maxbounds, size=(140, 4))  # neighbour nodes + trajectory segments (35, 4, 8) 35 neighbour nodes, 4 trajectory points, 8 dimensions

# Number of evaluation
num_times = 1e4

suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M')

# Evaluation
profiler = Profiler()
profiler.start()

for i in range(int(num_times)):
    grid.get_values(brt4d, neighbours)

profiler.stop()
profiler.print()
# print(profiler.output_text(unicode=True, color=True))
# Save as HTML
with open(f'paper_examples/hj_nav/results/brt4d_{num_times}_evaluations_{suffix}.html', 'w') as f:
    f.write(profiler.output_html())
