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

def world_to_map(xy, originx, originy, resolution):
    origin_xy = np.array([originx, originy])
    cell_fraction_xy = (xy -  origin_xy) / resolution
    
    return np.round(cell_fraction_xy).astype(int)

# Dynamics bounds: [x, y, v, theta]
minbounds = np.array([-6.5, -13.0, -0.2, 0.0])
maxbounds = np.array([5.05, 15.05, 0.8, 2*np.pi])
# num_grids = np.array([231, 150, 50, 50])

# Costmap loading
data = np.load("paper_examples/hj_nav/data/static_costmap_and_info.npz")
costmap2d = data["costmap"]
costmap_info = {
    'origin': [-6.5, -13.0],
    'resolution': 0.05000000074505806,
    'width': 231,
    'height': 561
}
print(costmap2d.shape)  # (561, 231)

neighbours_world = np.random.uniform(low=minbounds, high=maxbounds, size=(140, 4))[:, :2]  # neighbour nodes and trajectory segments (35, 4, 8) 35 neighbour nodes, 4 trajectory points, 8 dimensions

neighbours_map = world_to_map(neighbours_world, costmap_info["origin"][0], costmap_info["origin"][1], costmap_info["resolution"])
y, x = neighbours_map[:, 0], neighbours_map[:, 1]
# print(neighbours_map.shape)
# print(y.shape)

# Number of evaluation
num_times = 1e4

suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M')

# Evaluation
profiler = Profiler()
profiler.start()

for i in range(int(num_times)):
    neighbours_map = world_to_map(neighbours_world, costmap_info["origin"][0], costmap_info["origin"][1], costmap_info["resolution"])
    y, x = neighbours_map[:, 0], neighbours_map[:, 1]
    costmap2d[x, y]

profiler.stop()
profiler.print()
# print(profiler.output_text(unicode=True, color=True))
# Save as HTML
with open(f'paper_examples/hj_nav/results/obs2d_{num_times}_evaluations_{suffix}.html', 'w') as f:
    f.write(profiler.output_html())
