import imp
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
import math

from pyinstrument import Profiler

# Define grid
g = Grid(np.array([-10.0, -10.0, -0.1, -math.pi]), np.array([10.0, 10.0, 0.8, math.pi]), 4, np.array([231, 150, 50, 50]), [3])

# Define my object
my_car = DubinsCar4D(uMode="max", dMode="min")

# Use the grid to initialize initial value function
Initial_value_f = CylinderShape(g, [2,3], np.array([0., 2., 0., 0.]), 0.8)

# Look-back length and time step
lookback_length = 1.5
t_step = 0.05

small_number = 1e-5

tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

po = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1,3],
                  slicesCut=[5], save_fig=True, filename="test_obs_avoid.png")

# In this example, we compute a Backward Reachable Tube
compMethods = { "TargetSetMode": "minVWithV0"}
# result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, po, saveAllTimeSteps=False, accuracy="medium")
# np.save(f"/localhome/hha160/projects/optimized_dp/paper_examples/Dubins4D_value_function.npy", result)

result = np.load("/localhome/hha160/projects/optimized_dp/paper_examples/Dubins4D_value_function.npy")

# Record the indexing time
neighbours = np.random.rand(64, 4)
profiler = Profiler()
profiler.start()

for i in range(1000):
    g.get_values(result, neighbours)

profiler.stop()
profiler.print()
# print(profiler.output_text(unicode=True, color=True))
# Save as HTML
with open('profile_output.html', 'w') as f:
    f.write(profiler.output_html())
