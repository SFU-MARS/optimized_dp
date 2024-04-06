import heterocl as hcl
import numpy as np
import time

from odp.Plots import plot_isosurface

# Backward reachable set computation library
from odp.computeGraphs import graph_2D, graph_3D, graph_4D, graph_5D, graph_6D, graph_7D, graph_8D
from odp.TimeToReach import TTR_3D, TTR_4D, TTR_5D 

# Value Iteration library
from odp.valueIteration import value_iteration_3D, value_iteration_4D, value_iteration_5D, value_iteration_6D
import os, psutil

def solveValueIteration(MDP_obj):
    print("Welcome to optimized_dp \n")
    # Initialize the HCL environment
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

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
             plot_option, saveAllTimeSteps=False,
             accuracy="low", untilConvergent=False, epsilon=2e-3):

    # print("Welcome to optimized_dp \n")
    if type(multiple_value) == list:
        # We have both goal and obstacle set
        target = multiple_value[0] # Target set
        constraint = multiple_value[1] # Obstacle set
    else:
        target = multiple_value
        constraint = None
    
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    ################# INITIALIZE DATA TO BE INPUT INTO EXECUTABLE ##########################

    print("Initializing\n")

    if constraint is None:
        print("No obstacles set !")
        init_value = target
    else: 
        print("Obstacles set exists !")
        constraint_dim = constraint.ndim

        # Time-varying obstacle sets
        if constraint_dim > grid.dims:
            constraint_i = constraint[...,0]
        else:
            # Time-invariant obstacle set
            constraint_i = constraint

        init_value = np.maximum(target, -constraint_i)
        init_value = np.array(init_value, dtype='float32')

    process = psutil.Process(os.getpid())
    # print("Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

    # Tensors input to our computation graph
    V_0 = hcl.asarray(init_value)
    V_1 = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))

    process = psutil.Process(os.getpid())
    # print("Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

    # Check which target set or initial value set
    if compMethod["TargetSetMode"] != "minVWithVTarget" and compMethod["TargetSetMode"] != "maxVWithVTarget":
        l0 = hcl.asarray(init_value)
    else:
        l0 = hcl.asarray(target)

    del init_value
    # For debugging purposes
    if grid.dims == 4:
        probe = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))

    process = psutil.Process(os.getpid())
    # print("Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

    # Array for each state values
    list_x1 = np.reshape(grid.vs[0], grid.pts_each_dim[0])
    list_x2 = np.reshape(grid.vs[1], grid.pts_each_dim[1])
    if grid.dims >= 3:
        list_x3 = np.reshape(grid.vs[2], grid.pts_each_dim[2])
    if grid.dims >= 4:
        list_x4 = np.reshape(grid.vs[3], grid.pts_each_dim[3])
    if grid.dims >= 5:
        list_x5 = np.reshape(grid.vs[4], grid.pts_each_dim[4])
    if grid.dims >= 6:
        list_x6 = np.reshape(grid.vs[5], grid.pts_each_dim[5])
    if grid.dims >= 7:
        list_x7 = np.reshape(grid.vs[6], grid.pts_each_dim[6])
    if grid.dims >= 8:
        list_x8 = np.reshape(grid.vs[7], grid.pts_each_dim[7])

    # Convert state arrays to hcl array type
    list_x1 = hcl.asarray(list_x1)
    list_x2 = hcl.asarray(list_x2)
    if grid.dims >= 3:
        list_x3 = hcl.asarray(list_x3)
    if grid.dims >= 4:
        list_x4 = hcl.asarray(list_x4)
    if grid.dims >= 5:
        list_x5 = hcl.asarray(list_x5)
    if grid.dims >= 6:
        list_x6 = hcl.asarray(list_x6)
    if grid.dims >= 7:
        list_x7 = hcl.asarray(list_x7)
    if grid.dims >= 8:
        list_x8 = hcl.asarray(list_x8)

    # Get executable, obstacle check intial value function
    if grid.dims == 2:
        solve_pde = graph_2D(dynamics_obj, grid, compMethod["TargetSetMode"], accuracy)

    if grid.dims == 3:
        solve_pde = graph_3D(dynamics_obj, grid, compMethod["TargetSetMode"], accuracy)

    if grid.dims == 4:
        solve_pde = graph_4D(dynamics_obj, grid, compMethod["TargetSetMode"], accuracy)

    if grid.dims == 5:
        solve_pde = graph_5D(dynamics_obj, grid, compMethod["TargetSetMode"], accuracy)

    if grid.dims == 6:
        solve_pde = graph_6D(dynamics_obj, grid, compMethod["TargetSetMode"], accuracy)
    
    if grid.dims == 7:
        solve_pde = graph_7D(dynamics_obj, grid, compMethod["TargetSetMode"], accuracy)
    
    if grid.dims == 8:
        solve_pde = graph_8D(dynamics_obj, grid, compMethod["TargetSetMode"], accuracy)

    """ Be careful, for high-dimensional array (5D or higher), saving value arrays at all the time steps may 
    cause your computer to run out of memory """
    if saveAllTimeSteps is True:
        valfuncs = np.zeros(np.insert(tuple(grid.pts_each_dim), grid.dims, len(tau)))
        valfuncs[..., -1 ] = V_0.asnumpy()
        print(valfuncs.shape)


    ################ USE THE EXECUTABLE ############
    # Variables used for timing
    execution_time = 0
    iter = 0
    tNow = tau[0]
    print("Started running\n")

    process = psutil.Process(os.getpid())
    # print("Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

    # Backward reachable set/tube will be computed over the specified time horizon
    # Or until convergent ( which ever happens first )
    for i in range (1, len(tau)):
        #tNow = tau[i-1]
        t_minh= hcl.asarray(np.array((tNow, tau[i])))
        
        # taking obstacle at each timestep
        if "ObstacleSetMode" in compMethod and constraint_dim > grid.dims:
            constraint_i = constraint[...,i]

        while tNow <= tau[i] - 1e-4:
            prev_arr = V_0.asnumpy()
            # Start timing
            iter += 1
            start = time.time()

            # Run the execution and pass input into graph
            if grid.dims == 2:
                solve_pde(V_1, V_0, list_x1, list_x2, t_minh, l0)
            if grid.dims == 3:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, t_minh, l0)
            if grid.dims == 4:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, t_minh, l0, probe)
            if grid.dims == 5:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5 ,t_minh, l0)
            if grid.dims == 6:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, t_minh, l0)
            if grid.dims == 7:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, list_x7, t_minh, l0)
            if grid.dims == 8:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, list_x7, list_x8, t_minh, l0)

            tNow = t_minh.asnumpy()[0]
            process = psutil.Process(os.getpid())
            # print("Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

            # Calculate computation time
            execution_time += time.time() - start

            # If ObstacleSetMode is specified by user
            if "ObstacleSetMode" in compMethod:
                if compMethod["ObstacleSetMode"] == "maxVWithObstacle":
                    tmp_val = np.maximum(V_0.asnumpy(), -constraint_i)
                elif compMethod["ObstacleSetMode"] == "minVWithObstacle":
                    tmp_val = np.minimum(V_0.asnumpy(), -constraint_i)

                # Update final result
                V_1 = hcl.asarray(tmp_val)
                # Update input for next iteration
                V_0 = hcl.asarray(tmp_val)

            # Some information printin
            print(t_minh)
            print("Computational time to integrate (s): {:.5f}".format(time.time() - start))
            process = psutil.Process(os.getpid())
            # print("Gigabytes consumed {}".format(process.memory_info().rss/1e9))  # in bytes

            if untilConvergent is True:
                # Compare difference between V_{t-1} and V_{t} and choose the max changes
                diff = np.amax(np.abs(V_1.asnumpy() - prev_arr))
                print("Max difference between V_old and V_new : {:.5f}".format(diff))
                if diff < epsilon:
                    print("Result converged ! Exiting the compute loop. Have a good day.")
                    break
        else: # if it didn't break because of convergent condition
            if saveAllTimeSteps is True:
                valfuncs[..., -1-i] = V_1.asnumpy()
            continue
        break # only if convergent condition is achieved


    # Time info printing
    print("Total kernel time (s): {:.5f}".format(execution_time))
    print("Finished solving\n")

    ##################### PLOTTING #####################
    if plot_option.do_plot :
        # Only plots last value array for now
        plot_isosurface(grid, V_1.asnumpy(), plot_option)

    if saveAllTimeSteps is True:
        valfuncs[..., 0] = V_1.asnumpy()
        return valfuncs

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

def computeSpatDerivArray(grid, V, deriv_dim, accuracy="low"):
    # Return a tensor same size as V that contains spatial derivatives at every state in V
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    # Need to make sure that value array has the same size as grid
    assert list(V.shape) == list(grid.pts_each_dim)

    V_0 = hcl.asarray(V)
    spatial_deriv = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))

    # Get executable, obstacle check intial value function
    if grid.dims == 2:
        compute_SpatDeriv = graph_2D(None, grid, "None", accuracy,
                                     generate_SpatDeriv=True, deriv_dim=deriv_dim)
    if grid.dims == 3:
        compute_SpatDeriv = graph_3D(None, grid, "None", accuracy,
                                     generate_SpatDeriv=True, deriv_dim=deriv_dim)
    if grid.dims == 4:
        compute_SpatDeriv = graph_4D(None, grid, "None", accuracy,
                                     generate_SpatDeriv=True, deriv_dim=deriv_dim)
    if grid.dims == 5:
        compute_SpatDeriv = graph_5D(None, grid, "None", accuracy,
                                     generate_SpatDeriv=True, deriv_dim=deriv_dim)
    if grid.dims == 6:
        compute_SpatDeriv = graph_6D(None, grid, "None", accuracy,
                                     generate_SpatDeriv=True, deriv_dim=deriv_dim)
    if grid.dims == 8:
        compute_SpatDeriv = graph_8D(None, grid, "None", accuracy,
                                     generate_SpatDeriv=True, deriv_dim=deriv_dim)

    compute_SpatDeriv(V_0, spatial_deriv)
    return spatial_deriv.asnumpy()
