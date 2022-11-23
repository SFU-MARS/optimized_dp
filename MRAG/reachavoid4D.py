import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import heterocl as hcl
import math
import numpy as np
import plotly.graph_objects as go
from odp.Grid import Grid
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics.AttackerDefender4D import AttackerDefender4D
# Plot options
from odp.Plots import PlotOptions
from odp.Plots.plotting_utilities import plot_2d
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import math

'''
# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCapture
from odp.dynamics import DubinsCar4D2
'''
from odp.Plots import PlotOptions
from odp.solver import HJSolver
from odp.Plots.plotting_utilities import *
#from odp.Plots.plot_contour_5D import *

class ReachAvoid:
    '''
    Agent1 : Attacker
    Agent2 : Defender
    '''
    def __init__(self, x = [0,0,0,0], uMax = 1, uMin = -1, dMax = 1, dMin = -1, uMode = 'min', dMode = 'max', speed_a = 1, speed_b = 1):
        self.x = x
        self.uMax = uMax
        self.uMin = uMin

        self.dMax = dMax
        self.dMin = dMin
        # QUESTION: uMode and dMode for reach-avoid?
        self.uMode = uMode
        self.dMode = dMode
        # max speed for attacker and defender 
        self.speed_a = speed_a
        self.speed_b = speed_b

    def dynamics(self, t, state, uOpt, dOpt):
        x1_dot = hcl.scalar(0, "x_1_dot")
        x2_dot = hcl.scalar(0, "x_2_dot")
        x3_dot = hcl.scalar(0, "x_3_dot")
        x4_dot = hcl.scalar(0, "x_4_dot")
       
        x1_dot[0] = self.speed_a * uOpt[0]
        x2_dot[0] = self.speed_a * uOpt[1]
        x3_dot[0] = self.speed_b * dOpt[0]
        x4_dot[0] = self.speed_b * dOpt[1]
       
        return (x1_dot[0], x2_dot[0], x3_dot[0], x4_dot[0])

    def opt_ctrl(self, t, state, spat_deriv):
        # optimal control is (x1_dot, x2_dot), the direction to multiply by the scalar
        uOpt_x = hcl.scalar(0, "u")
        uOpt_y = hcl.scalar(0, "a")
        # Just create and pass back, even though they're not used
        in3   = hcl.scalar(0, "in3")
        in4   = hcl.scalar(0, "in4")
        # declare the hcl scalars for relevant spat_derivs
        deriv1 = hcl.scalar(0, "deriv1")
        deriv2 = hcl.scalar(0, "deriv2")
        deriv1[0] = spat_deriv[0]
        deriv2[0] = spat_deriv[1]
        ctrl_len = hcl.sqrt(deriv1[0] * deriv1[0] + deriv2[0] * deriv2[0])
        if self.uMode == "min":
            uOpt_x[0] = -1.0 * deriv1[0] / ctrl_len
            uOpt_y[0] = -1.0 * deriv2[0] / ctrl_len
        else:
            uOpt_x[0] = deriv1[0]/ ctrl_len
            uOpt_y[0] = deriv2[0] / ctrl_len

        return (uOpt_x[0], uOpt_y[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        dOpt_x = hcl.scalar(0, "d_x")
        dOpt_y = hcl.scalar(0, "d_y")
        # Just create and pass back, even though they're not used
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")

        deriv1 = hcl.scalar(0, "deriv1")
        deriv2 = hcl.scalar(0, "deriv2")
        deriv1[0] = spat_deriv[2] 
        deriv2[0] = spat_deriv[3] 
        dstb_len = hcl.sqrt(deriv1[0] * deriv1[0] + deriv2[0] * deriv2[0])
      
        if self.dMode == "max":
            
            dOpt_x[0] = deriv1[0] / dstb_len
            dOpt_y[0] = deriv2[0] / dstb_len
        else:
           
            dOpt_x[0] = -1 * deriv1[0]/ dstb_len
            dOpt_y[0] = -1 * deriv2[0] / dstb_len
           
        return (dOpt_x[0], dOpt_y[0], d3[0], d4[0]) 
    
    def capture_set(self, grid, capture_radius, mode):
        
        data = np.zeros(grid.pts_each_dim)

        data = data + np.power(grid.vs[0] - grid.vs[2], 2)
        data = data + np.power(grid.vs[1] - grid.vs[3], 2)
       # data = np.sqrt(data) - radius
        if mode == "capture":
            return np.sqrt(data) - capture_radius
        if mode == "escape":
            return capture_radius - np.sqrt(data)
           
        

def main():
    g = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([31, 31, 31, 31]))
    # Initialize the dynamics. We assume the attackers' perspective.
    attacker_perpective = ReachAvoid(uMode='min', dMode='max')

    # static obstacles from the baseline paper
    obs1 = ShapeRectangle(g, [-0.1, -1.0, -1000, -1000], [0.1, -0.3, 1000, 1000])  # attacker stuck in obs1
    obs2 = ShapeRectangle(g, [-0.1, 0.30, -1000, -1000], [0.1, 0.60, 1000, 1000])  # attacker stuck in obs2
    obstacle = np.minimum(obs1, obs2)

    obs1_reach = ShapeRectangle(g, [-1000, -1000, -0.1, -1.0], [1000, 1000, 0.1, -0.3])  # defender stuck in obs1
    obs2_reach = ShapeRectangle(g, [-1000, -1000, -0.1, 0.30], [1000, 1000, 0.1, 0.60])  # defender stuck in obs2
    obstacle_reach = np.minimum(obs1_reach, obs2_reach)
    
    goal_a = ShapeRectangle(g, [0.6, 0.1, -1000, -1000], [0.8, 0.3, 1000, 1000])  # attacker arrives target

    capture_region = attacker_perpective.capture_set(g, 0.3, "capture")
    escape_region = attacker_perpective.capture_set(g, 0.3, "escape")
    #reach = np.maximum(goal_a, escape_region)
    reach = np.minimum(np.maximum(goal_a, escape_region), obstacle_reach)
    avoid = np.minimum(obstacle, capture_region)

    # po1 = PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0,1,2],
    #               slicesCut=[3])
   # plot_isosurface(g, avoid, po1) #plot the reach and avoid sets for debugging.
   # plot_isosurface(g, reach, po1)

    lookback_length = 5.0
    t_step = 0.05
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    # Get our first safety bubble.
    po2 = PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[2])
    compMethods = { "TargetSetMode": "minVWithVTarget", "ObstacleSetMode": "maxVWithObstacle"}
                
    # HJSolver(dynamics object, g, initial value function, time length, system objectives, plotting options)
    result = HJSolver(attacker_perpective, g, [reach, avoid], tau, compMethods, po2, saveAllTimeSteps=True)
    np.save('result.npy', result)
    print(result.shape)
    # print(result[0], result[30][30][30][30], result[15][15][15][15], result[15][10][15][10], result[0][15][1][15])

    

    x1_idx = [0, 10, 20, 30]
    x2_idx = [0, 10, 20, 30]
    #x4_idx = [0, 10, 20, 30]
# loop over these indices in the g for fixed values
#     for idx1 in x1_idx:
#         for idx2 in x2_idx:
#
#             plot_reachavoid(result, [idx1, idx2], 2)

    return

if __name__ == "__main__":
    
    main()
  
