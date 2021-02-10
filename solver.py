import heterocl as hcl
import numpy as np
import time

from Plots.plotting_utilities import *
from argparse import ArgumentParser
from computeGraphs.graph_3D import *
from computeGraphs.graph_4D import *
from computeGraphs.graph_5D import *
from computeGraphs.graph_6D import *

def HJSolver(dynamics_obj, grid, init_value, tau, compMethod, plot_option):
    print("Welcome to optimized_dp \n")

    ################### PARSING ARGUMENTS FROM USERS #####################

    parser = ArgumentParser()
    parser.add_argument("-p", "--plot", default=True, type=bool)
    # # Print out LLVM option only
    # parser.add_argument("-l", "--llvm", default=False, type=bool)
    args = parser.parse_args()

    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    ################# INITIALIZE DATA TO BE INPUT INTO EXECUTABLE ##########################

    print("Initializing\n")

    V_0 = hcl.asarray(init_value)
    V_1 = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))
    l0  = hcl.asarray(init_value)
    probe = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))
    #obstacle = hcl.asarray(cstraint_values)

    list_x1 = np.reshape(grid.vs[0], grid.pts_each_dim[0])
    list_x2 = np.reshape(grid.vs[1], grid.pts_each_dim[1])
    list_x3 = np.reshape(grid.vs[2], grid.pts_each_dim[2])
    if grid.dims >= 4:
        list_x4 = np.reshape(grid.vs[3], grid.pts_each_dim[3])
    if grid.dims >= 5:
        list_x5 = np.reshape(grid.vs[4], grid.pts_each_dim[4])
    if grid.dims >= 6:
        list_x6 = np.reshape(grid.vs[5], grid.pts_each_dim[5])


    # Convert to hcl array type
    list_x1 = hcl.asarray(list_x1)
    list_x2 = hcl.asarray(list_x2)
    list_x3 = hcl.asarray(list_x3)
    if grid.dims >= 4:
        list_x4 = hcl.asarray(list_x4)
    if grid.dims >= 5:
        list_x5 = hcl.asarray(list_x5)
    if grid.dims >= 6:
        list_x6 = hcl.asarray(list_x6)

    # Get executable
    if grid.dims == 3:
        solve_pde = graph_3D(dynamics_obj, grid, compMethod)
    if grid.dims == 4:
        solve_pde = graph_4D(dynamics_obj, grid, compMethod)
    if grid.dims == 5:
        solve_pde = graph_5D(dynamics_obj, grid, compMethod)
    if grid.dims == 6:
        solve_pde = graph_6D(dynamics_obj, grid, compMethod)

    # Print out code for different backend
    #print(solve_pde)

    ################ USE THE EXECUTABLE ############
    # Variables used for timing
    execution_time = 0
    iter = 0
    tNow = tau[0]
    for i in range (1, len(tau)):
        #tNow = tau[i-1]
        t_minh= hcl.asarray(np.array((tNow, tau[i])))
        while tNow <= tau[i] - 1e-4:
             tmp_arr = V_0.asnumpy()
             # Start timing
             iter += 1
             start = time.time()

             print("Started running\n")

             # Run the execution and pass input into graph
             if grid.dims == 3:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, t_minh, l0)
             if grid.dims == 4:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, t_minh, l0, probe)
             if grid.dims == 5:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5 ,t_minh, l0)
             if grid.dims == 6:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, t_minh, l0)

             tNow = np.asscalar((t_minh.asnumpy())[0])

             # Calculate computation time
             execution_time += time.time() - start

             # Some information printing
             print(t_minh)
             print("Computational time to integrate (s): {:.5f}".format(time.time() - start))


    # Time info printing
    print("Total kernel time (s): {:.5f}".format(execution_time))
    print("Finished solving\n")

    ##################### PLOTTING #####################
    if args.plot:
        # plot Value table when speed is maximum
        plot_isosurface(grid, V_1.asnumpy(), plot_option)
        #plot_isosurface(g, my_V, [0, 1, 3], 10)
    return V_1.asnumpy()

