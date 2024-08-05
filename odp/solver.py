import heterocl as hcl
import numpy as np
import time

from odp.Plots import plot_isosurface, plot_valuefunction

# Backward reachable set computation library
from odp.computeGraphs import graph_1D, graph_2D, graph_3D, graph_4D, graph_5D, graph_6D
from odp.TimeToReach import TTR_2D, TTR_3D, TTR_4D, TTR_5D 

# Value Iteration library
from odp.valueIteration import value_iteration_3D, value_iteration_4D, value_iteration_5D, value_iteration_6D

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
        iter = 0
        resweep = 1
        while iter < MDP_obj._maxIters and resweep == 1:
            reSweep = hcl.asarray(np.zeros([1]))
            print("Currently at iteration {}".format(iter))
            f(V_opt, actions, intermeds, trans, interpV, gamma, epsilon, reSweep, iVals, sVals, bounds, goal, ptsEachDim, count,
                maxIters, useNN, fillVal)
            iter += 1
            resweep = reSweep.asnumpy()[0]
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
             accuracy="medium", untilConvergent=False, epsilon=2e-3):

    print("Welcome to optimized_dp \n")
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

    # Tensors input to our computation graph
    V_t = hcl.asarray(init_value)
    Hamiltonian = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))
    delta_t = hcl.asarray(np.zeros(1))
    
    # Check which target set or initial value set
    if compMethod["TargetSetMode"] != "minVWithVTarget" and compMethod["TargetSetMode"] != "maxVWithVTarget":
        l0 = hcl.asarray(init_value)
    else:
        l0 = hcl.asarray(target)

    # For debugging purposes
    #probe = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))

    # Array for each state values
    list_x1 = np.reshape(grid.vs[0], grid.pts_each_dim[0])
    if grid.dims >= 2:
        list_x2 = np.reshape(grid.vs[1], grid.pts_each_dim[1])
    if grid.dims >= 3:
        list_x3 = np.reshape(grid.vs[2], grid.pts_each_dim[2])
    if grid.dims >= 4:
        list_x4 = np.reshape(grid.vs[3], grid.pts_each_dim[3])
    if grid.dims >= 5:
        list_x5 = np.reshape(grid.vs[4], grid.pts_each_dim[4])
    if grid.dims >= 6:
        list_x6 = np.reshape(grid.vs[5], grid.pts_each_dim[5])

    # Convert state arrays to hcl array type
    list_x1 = hcl.asarray(list_x1)
    if grid.dims >= 2:
        list_x2 = hcl.asarray(list_x2)
    if grid.dims >= 3:
        list_x3 = hcl.asarray(list_x3)
    if grid.dims >= 4:
        list_x4 = hcl.asarray(list_x4)
    if grid.dims >= 5:
        list_x5 = hcl.asarray(list_x5)
    if grid.dims >= 6:
        list_x6 = hcl.asarray(list_x6)

    # Get executable, obstacle check intial value function
    if grid.dims == 1:
        solve_pde = graph_1D(dynamics_obj, grid, compMethod["TargetSetMode"], accuracy)

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

    """ Be careful, for high-dimensional array (5D or higher), saving value arrays at all the time steps may 
    cause your computer to run out of memory """
    if saveAllTimeSteps is True:
        valfuncs = np.zeros(np.insert(tuple(grid.pts_each_dim), grid.dims, len(tau)))
        valfuncs[..., -1 ] = V_t.asnumpy()
        print(valfuncs.shape)


    ################ USE THE EXECUTABLE ############
    # Variables used for timing
    execution_time = 0
    iter = 0
    tNow = tau[0]
    print("Started running\n")

    # Backward reachable set/tube will be computed over the specified time horizon
    # Or until convergent ( which ever happens first )
    for i in range (1, len(tau)):
        t_minh= hcl.asarray(np.array((tNow, tau[i])))
        # taking obstacle at each timestep
        if "ObstacleSetMode" in compMethod and constraint_dim > grid.dims:
            constraint_i = constraint[...,i]

        while tNow <= tau[i] - 1e-4:
            prev_arr = V_t.asnumpy()
            # Start timing
            iter += 1
            start = time.time()
            
            t = hcl.asarray(np.array([tNow]))

            # Run the execution and pass input into graph
            if grid.dims == 1:
                solve_pde(Hamiltonian, V_t, list_x1, delta_t, t, l0)
            if grid.dims == 2:
                solve_pde(Hamiltonian, V_t, list_x1, list_x2, delta_t, t, l0)
            if grid.dims == 3:
                solve_pde(Hamiltonian, V_t, list_x1, list_x2, list_x3, delta_t, t, l0)
            if grid.dims == 4:
                solve_pde(Hamiltonian, V_t, list_x1, list_x2, list_x3, list_x4, delta_t, t, l0)
            if grid.dims == 5:
                solve_pde(Hamiltonian, V_t, list_x1, list_x2, list_x3, list_x4, list_x5, delta_t, t, l0)
            if grid.dims == 6:
                solve_pde(Hamiltonian, V_t, list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, delta_t, t, l0)
            
            # Smallest timestep to integrate according to CFL condition
            dt = delta_t.asnumpy()[0]

            # if stabilizing time step is more than the step to next time, choose the latter
            dt = min(tau[i] - tNow,  dt)

            # Integrate using delta_t, doing this once corresponds to first order TVD RK
            # TODO: In-place update for more memory efficiency
            V_tp1 = V_t.asnumpy() + Hamiltonian.asnumpy() *  dt
            
            # If accuracy medium, we will take second order TVD RK, which requires second order integration
            # TODO: make integration accuracy a separate option
            if accuracy == "medium":

                # Convert to hcl array type
                V_tp1 = hcl.asarray(V_tp1)  

                t1 = tNow  + dt
                t1 = hcl.asarray(np.array([t1]))
                
                
                # Compute phi at t = (n + 2)
                if grid.dims == 1:
                    solve_pde(Hamiltonian, V_tp1, list_x1, delta_t, t1, l0)
                if grid.dims == 2:
                    solve_pde(Hamiltonian, V_tp1, list_x1, list_x2, delta_t, t1 , l0)
                if grid.dims == 3:
                    solve_pde(Hamiltonian, V_tp1, list_x1, list_x2, list_x3, delta_t, t1, l0)
                if grid.dims == 4:
                    solve_pde(Hamiltonian, V_tp1, list_x1, list_x2, list_x3, list_x4, delta_t, t1, l0)
                if grid.dims == 5:
                    solve_pde(Hamiltonian, V_tp1, list_x1, list_x2, list_x3, list_x4, list_x5, delta_t, t1, l0)
                if grid.dims == 6:
                    solve_pde(Hamiltonian, V_tp1, list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, delta_t, t1, l0)

                # Integrate
                # TODO: In-place update for more memory efficiency
                V_tp2 = V_tp1.asnumpy() + Hamiltonian.asnumpy() *  dt
                V_tp1 = 0.5 * (V_t.asnumpy() + V_tp2)

            # Computational method
            if "TargetSetMode" in compMethod:
                if compMethod["TargetSetMode"] == 'maxVWithV0' or compMethod["TargetSetMode"] == 'maxVWithVTarget':
                    V_tp1 = np.maximum(V_tp1, l0.asnumpy())
                if compMethod["TargetSetMode"] == 'minVWithV0' or compMethod["TargetSetMode"] == 'minVWithVTarget':
                    V_tp1 = np.minimum(V_tp1, l0.asnumpy())
                if compMethod["TargetSetMode"] == 'minVOverTime':
                    V_tp1 = np.minimum(V_tp1, V_t.asnumpy())
                if compMethod["TargetSetMode"] == 'maxVOverTime':
                    V_tp1 = np.maximum(V_tp1, V_t.asnumpy())
            
            # If ObstacleSetMode is specified by user
            if "ObstacleSetMode" in compMethod:
                if compMethod["ObstacleSetMode"] == "maxVWithObstacle":
                    V_tp1 = np.maximum(V_tp1, -constraint_i)
                elif compMethod["ObstacleSetMode"] == "minVWithObstacle":
                    V_tp1 = np.minimum(V_tp1, -constraint_i)

            # Copy over for input next iterations
            V_t = hcl.asarray(V_tp1)
            # Convert new V back into heterocl type
            Hamiltonian = hcl.asarray(Hamiltonian)

            # Increment the current time
            tNow += dt
            
            # Calculate computation time
            execution_time += time.time() - start

            # Some information printin
            print(np.array([tNow, tau[i]]))
            print("Computational time to integrate (s): {:.5f}".format(time.time() - start))

            if untilConvergent is True:
                # Compare difference between V_{t-1} and V_{t} and choose the max changes
                diff = np.amax(np.abs(V_t.asnumpy() - prev_arr))
                print("Max difference between V_old and V_new : {:.5f}".format(diff))
                if diff < epsilon:
                    print("Result converged ! Exiting the compute loop. Have a good day.")
                    break
        else: # if it didn't break because of convergent condition
            if saveAllTimeSteps is True:
                valfuncs[..., -1-i] = V_t.asnumpy()
            continue
        break # only if convergent condition is achieved


    # Time info printing
    print("Total kernel time (s): {:.5f}".format(execution_time))
    print("Finished solving\n")

    ##################### PLOTTING #####################
    if plot_option.do_plot :
        # Only plots last value array for now
        if plot_option.plot_type == "set":
            if saveAllTimeSteps is True:
                plot_isosurface(grid, valfuncs, plot_option)
            else:
                plot_isosurface(grid, V_t.asnumpy(), plot_option)
        elif plot_option.plot_type == "value":
            if saveAllTimeSteps is True:
                plot_valuefunction(grid, valfuncs, plot_option)
            else:
                plot_valuefunction(grid,V_t.asnumpy(), plot_option)

    if saveAllTimeSteps is True:
        valfuncs[..., 0] = V_t.asnumpy()
        return valfuncs

    return V_t.asnumpy()

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
    if grid.dims >= 2:
        list_x2 = np.reshape(grid.vs[1], grid.pts_each_dim[1])
    if grid.dims >= 3:
        list_x3 = np.reshape(grid.vs[2], grid.pts_each_dim[2])
    if grid.dims >= 4:
        list_x4 = np.reshape(grid.vs[3], grid.pts_each_dim[3])
    if grid.dims >= 5:
        list_x5 = np.reshape(grid.vs[4], grid.pts_each_dim[4])
    if grid.dims >= 6:
        list_x6 = np.reshape(grid.vs[5], grid.pts_each_dim[5])

    # Convert states vector to hcl array type
    list_x1 = hcl.asarray(list_x1)
    if grid.dims >= 2:
        list_x2 = hcl.asarray(list_x2)
    if grid.dims >= 3:
        list_x3 = hcl.asarray(list_x3)
    if grid.dims >= 4:
        list_x4 = hcl.asarray(list_x4)
    if grid.dims >= 5:
        list_x5 = hcl.asarray(list_x5)
    if grid.dims >= 6:
        list_x6 = hcl.asarray(list_x6)

    # Get executable
    # if grid.dims == 1:
    #     solve_TTR = TTR_1D(dynamics_obj, grid)
    if grid.dims == 2:
        solve_TTR = TTR_2D(dynamics_obj, grid)
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
        if grid.dims == 1:
            solve_TTR(V_0, list_x1)
        if grid.dims == 2:
            solve_TTR(V_0, list_x1, list_x2)
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
    if plot_option.do_plot :
        # Only plots last value array for now
        if plot_option.plot_type == "set":
            plot_isosurface(grid, V_0.asnumpy(), plot_option)
        elif plot_option.plot_type == "value":
            plot_valuefunction(grid, V_0.asnumpy(), plot_option)

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
    if grid.dims == 1:
        compute_SpatDeriv = graph_1D(None, grid, "None", accuracy,
                                     generate_SpatDeriv=True, deriv_dim=deriv_dim)
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

    compute_SpatDeriv(V_0, spatial_deriv)
    return spatial_deriv.asnumpy()
