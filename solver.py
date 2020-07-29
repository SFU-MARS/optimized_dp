import heterocl as hcl
import numpy as np
import time
#import plotly.graph_objects as go

from computeGraphs.CustomGraphFunctions import *
from Plots.plotting_utilities import *
from user_definer import *
from argparse import ArgumentParser
from computeGraphs.graph_4D import *
from computeGraphs.graph_5D import *
from computeGraphs.graph_6D import *
import scipy.io as sio

import math


def main():
    ################### PARSING ARGUMENTS FROM USERS #####################

    parser = ArgumentParser()
    parser.add_argument("-p", "--plot", default=True, type=bool)
    # Print out LLVM option only
    parser.add_argument("-l", "--llvm", default=False, type=bool)
    args = parser.parse_args()

    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    ################# INITIALIZE DATA TO BE INPUT INTO EXECUTABLE ##########################

    print("Initializing\n")

    V_0 = hcl.asarray(my_shape)
    V_1 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    l0  = hcl.asarray(my_shape)
    #probe = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    #obstacle = hcl.asarray(cstraint_values)

    list_x1 = np.reshape(g.vs[0], g.pts_each_dim[0])
    list_x2 = np.reshape(g.vs[1], g.pts_each_dim[1])
    list_x3 = np.reshape(g.vs[2], g.pts_each_dim[2])
    if g.dims >= 4:
        list_x4 = np.reshape(g.vs[3], g.pts_each_dim[3])
    if g.dims >= 5:
        list_x5 = np.reshape(g.vs[4], g.pts_each_dim[4])
    if g.dims >= 6:
        list_x6 = np.reshape(g.vs[5], g.pts_each_dim[5])


    # Convert to hcl array type
    list_x1 = hcl.asarray(list_x1)
    list_x2 = hcl.asarray(list_x2)
    list_x3 = hcl.asarray(list_x3)
    if g.dims >= 4:
        list_x4 = hcl.asarray(list_x4)
    if g.dims >= 5:
        list_x5 = hcl.asarray(list_x5)
    if g.dims >= 6:
        list_x6 = hcl.asarray(list_x6)

    # Get executable
    if g.dims == 4:
        solve_pde = graph_4D()
    if g.dims == 5:
        solve_pde = graph_5D()
    if g.dims == 6:
        solve_pde = graph_6D()

    # Print out code for different backend
    #print(solve_pde)

    ################ USE THE EXECUTABLE ############
    # Variables used for timing
    execution_time = 0
    lookback_time = 0

    tNow = tau[0]
    for i in range (1, len(tau)):
        #tNow = tau[i-1]
        t_minh= hcl.asarray(np.array((tNow, tau[i])))
        while tNow <= tau[i] - 1e-4:
             # Start timing
             start = time.time()

             print("Started running\n")

             # Run the execution and pass input into graph
             if g.dims == 4:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, t_minh, l0)
             if g.dims == 5:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5 ,t_minh, l0)
             if g.dims == 6:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, t_minh, l0)

             tNow = np.asscalar((t_minh.asnumpy())[0])

             # Calculate computation time
             execution_time += time.time() - start

             # Some information printing
             print(t_minh)
             print("Computational time to integrate (s): {:.5f}".format(time.time() - start))
             # Saving data into disk


    # Time info printing
    print("Total kernel time (s): {:.5f}".format(execution_time))
    print("Finished solving\n")

    # V1 is the final value array, fill in anything to use it



    ##################### PLOTTING #####################
    if args.plot:
        plot_isosurface(g, V_1.asnumpy(), [0, 1, 3])

if __name__ == '__main__':
  main()
