import heterocl as hcl
import numpy as np
import time

from Plots.plotting_utilities import *
from argparse import ArgumentParser

# Backward reachable set computation library
from computeGraphs.graph_3D import *
from computeGraphs.graph_4D import *
from computeGraphs.graph_5D import *
from computeGraphs.graph_6D import *

from TimeToReach.TimeToReach_3D import  *
from TimeToReach.TimeToReach_4D import  *
from TimeToReach.TimeToReach_5D import  *

# Value Iteration library
from valueIteration.value_iteration_3D import *
from valueIteration.value_iteration_4D import *
from valueIteration.value_iteration_5D import *
from valueIteration.value_iteration_6D import *

def solveValueIteration(MDP_obj):
    print("Welcome to optimized_dp \n")
    # Initialize the HCL environment
    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    ########################################## INITIALIZE ##########################################

    # Convert the python array to hcl type array
    V_opt = hcl.asarray(np.zeros(MDP_obj._ptsEachDim))
    intermeds = hcl.asarray(np.ones(MDP_obj._actions.shape[0]))
    trans = hcl.asarray(MDP_obj._trans)
    gamma = hcl.asarray(MDP_obj._gamma)
    epsilon = hcl.asarray(MDP_obj._epsilon)
    count = hcl.asarray(np.zeros(1))
    maxIters = hcl.asarray(MDP_obj._maxIters)
    actions = hcl.asarray(MDP_obj._actions)
    bounds = hcl.asarray(MDP_obj._bounds)
    goal = hcl.asarray(MDP_obj._goal)
    ptsEachDim = hcl.asarray(MDP_obj._ptsEachDim)
    sVals = hcl.asarray(np.zeros([MDP_obj._bounds.shape[0]]))
    iVals = hcl.asarray(np.zeros([MDP_obj._bounds.shape[0]]))
    interpV = hcl.asarray(np.zeros([1]))
    useNN = hcl.asarray(MDP_obj._useNN)

    print(MDP_obj._bounds.shape[0])
    print(np.zeros([MDP_obj._bounds.shape[0]]))
    if MDP_obj._bounds.shape[0] == 3:
        fillVal = hcl.asarray(MDP_obj._fillVal)
        f = value_iteration_3D(MDP_obj)
    if MDP_obj._bounds.shape[0] == 4:
        f = value_iteration_4D(MDP_obj)
    if MDP_obj._bounds.shape[0] == 5:
        f = value_iteration_5D(MDP_obj)
    if MDP_obj._bounds.shape[0] == 6:
        f = value_iteration_6D(MDP_obj)

    # Build the graph and use the executable
    # Now use the executable
    t_s = time.time()
    if MDP_obj._bounds.shape[0] == 3:
        f(V_opt, actions, intermeds, trans, interpV, gamma, epsilon, iVals, sVals, bounds, goal, ptsEachDim, count,
            maxIters, useNN, fillVal)
    else:
        f(V_opt, actions, intermeds, trans, interpV, gamma, epsilon, iVals, sVals, bounds, goal, ptsEachDim, count,
          maxIters, useNN)
    t_e = time.time()

    V = V_opt.asnumpy()
    c = count.asnumpy()
    print("Finished in ", int(c[0]), " iterations")
    print("Took        ", t_e - t_s, " seconds")

    # # Write results to file
    # if (MDP_obj.dir_path):
    #     dir_path = MDP_obj.dir_path
    # else:
    #     dir_path = "./hcl_value_matrix_test/"
    #
    # if (MDP_obj.file_name):
    #     file_name = MDP_obj.file_name
    # else:
    #     file_name = "hcl_value_iteration_" + str(int(c[0])) + "_iterations_by" + (
    #         "_Interpolation" if MDP_obj._useNN[0] == 0 else "_NN")
    # MDP_obj.writeResults(V, dir_path, file_name, just_values=True)
    return V

def HJSolver(dynamics_obj, grid, multiple_value, tau, compMethod,
             plot_option, accuracy="low", save_all_t=False):
    print("Welcome to optimized_dp \n")
    if type(multiple_value) == list:
        init_value = multiple_value[0]
        constraint = multiple_value[1]
    else:
        init_value = multiple_value
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
    # TODO: change to use hcl api
    # add extra dimension to grid of length tau
    V_all_t = np.zeros(list(grid.pts_each_dim) + [len(tau)])
    V_all_t[..., 0] = init_value

    l0 = hcl.asarray(init_value)
    probe = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))

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
        solve_pde = graph_3D(dynamics_obj, grid, compMethod["PrevSetsMode"], accuracy)

    if grid.dims == 4:
        solve_pde = graph_4D(dynamics_obj, grid, compMethod["PrevSetsMode"], accuracy)

    if grid.dims == 5:
        solve_pde = graph_5D(dynamics_obj, grid, compMethod["PrevSetsMode"], accuracy)

    if grid.dims == 6:
        solve_pde = graph_6D(dynamics_obj, grid, compMethod["PrevSetsMode"], accuracy)

    # Print out code for different backend
    #print(solve_pde)

    ################ USE THE EXECUTABLE ############
    # Variables used for timing
    execution_time = 0
    iter = 0
    tNow = tau[0]
    print("Started running\n")
    for i in range (1, len(tau)):
        #tNow = tau[i-1]
        t_minh= hcl.asarray(np.array((tNow, tau[i])))
        print(i)
        while tNow <= tau[i] - 1e-4:
             tmp_arr = V_0.asnumpy()
             # Start timing
             iter += 1
             start = time.time()

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

             # If TargetSetMode is specified by user
             if "TargetSetMode" in compMethod:
                if compMethod["TargetSetMode"] == "max":
                    tmp_val = np.maximum(V_0.asnumpy(), constraint)
                elif compMethod["TargetSetMode"] == "min":
                    tmp_val = np.minimum(V_0.asnumpy(), constraint)
                # Update final result
                V_1 = hcl.asarray(tmp_val)
                # Update input for next iteration
                V_0 = hcl.asarray(tmp_val)

             # Some information printing
             print(t_minh)
             print("Computational time to integrate (s): {:.5f}".format(time.time() - start))
        # TODO: change to hcl api
        V_all_t[..., i] = V_1.asnumpy()

    # Time info printing
    print("Total kernel time (s): {:.5f}".format(execution_time))
    print("Finished solving\n")

    ##################### PLOTTING #####################
    if args.plot:
        # Only plots last value array for now
        plot_isosurface(grid, V_1.asnumpy(), plot_option)

    if save_all_t:
        return V_all_t

    return V_1.asnumpy()

def TTRSolver(dynamics_obj, grid, init_value, epsilon, plot_option):
    print("Welcome to optimized_dp \n")
    ################# INITIALIZE DATA TO BE INPUT INTO EXECUTABLE ##########################

    print("Initializing\n")
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    # Convert initial distance value function to initial time-to-reach value function
    init_value[init_value < 0] = 0
    init_value[init_value > 0] = 1000
    V_0 = hcl.asarray(init_value)
    prev_val = np.zeros(init_value.shape)

    # Re-shape states vector
    list_x1 = np.reshape(grid.vs[0], grid.pts_each_dim[0])
    list_x2 = np.reshape(grid.vs[1], grid.pts_each_dim[1])
    list_x3 = np.reshape(grid.vs[2], grid.pts_each_dim[2])
    if grid.dims >= 4:
        list_x4 = np.reshape(grid.vs[3], grid.pts_each_dim[3])
    if grid.dims >= 5:
        list_x5 = np.reshape(grid.vs[4], grid.pts_each_dim[4])
    if grid.dims >= 6:
        list_x6 = np.reshape(grid.vs[5], grid.pts_each_dim[5])

    # Convert states vector to hcl array type
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
        solve_TTR = TTR_3D(dynamics_obj, grid)
    if grid.dims == 4:
        solve_TTR = TTR_4D(dynamics_obj, grid)
    if grid.dims == 5:
        solve_TTR = TTR_5D(dynamics_obj, grid)
    if grid.dims == 6:
        solve_TTR = TTR_6D(dynamics_obj, grid)
    print("Got Executable\n")

    # Print out code for different backend
    # print(solve_pde)

    ################ USE THE EXECUTABLE ############
    error = 10000
    count = 0
    start = time.time()
    while error > epsilon:
        print("Iteration: {} Error: {}".format(count, error))
        count += 1
        if grid.dims == 3:
            solve_TTR(V_0, list_x1, list_x2, list_x3)
        if grid.dims == 4:
            solve_TTR(V_0, list_x1, list_x2, list_x3, list_x4)
        if grid.dims == 5:
            solve_TTR(V_0, list_x1, list_x2, list_x3, list_x4, list_x5)
        if grid.dims == 6:
            solve_TTR(V_0, list_x1, list_x2, list_x3, list_x4, list_x5, list_x6 )

        error = np.max(np.abs(prev_val - V_0.asnumpy()))
        prev_val = V_0.asnumpy()
    print("Total TTR computation time (s): {:.5f}".format(time.time() - start))
    print("Finished solving\n")

    ##################### PLOTTING #####################
    plot_isosurface(grid, V_0.asnumpy(), plot_option)
    return V_0.asnumpy()

