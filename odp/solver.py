import heterocl as hcl
import numpy as np
import time

from odp.Plots import plot_isosurface, plot_valuefunction

# Backward reachable set computation library
from odp.computeGraphs import graph_1D, graph_2D, graph_3D, graph_4D, graph_5D, graph_6D, graph_7D, graph_8D
from odp.TimeToReach import TTR_1D, TTR_2D, TTR_3D, TTR_4D, TTR_5D, TTR_6D, TTR_7D, TTR_8D

# Value Iteration library
from odp.valueIteration import value_iteration_2D, value_iteration_3D, value_iteration_4D, value_iteration_5D, value_iteration_6D

# Executables for each number of dimensions
_GRAPHS = {1: graph_1D, 2: graph_2D, 3: graph_3D, 4: graph_4D, 5: graph_5D, 6: graph_6D, 7: graph_7D, 8: graph_8D}
_TTR_SOLVERS = {1: TTR_1D, 2: TTR_2D, 3: TTR_3D, 4: TTR_4D, 5: TTR_5D, 6: TTR_6D, 7: TTR_7D, 8: TTR_8D}
_VALUE_ITERATIONS = {2: value_iteration_2D, 3: value_iteration_3D, 4: value_iteration_4D,
                     5: value_iteration_5D, 6: value_iteration_6D}

def _state_arrays(grid):
    return [hcl.asarray(np.reshape(grid.vs[d], grid.pts_each_dim[d])) for d in range(grid.dims)]

def solveValueIteration(MDP_obj, grid, action_space,
                        gamma=0.9, epsilon=1e-3, maxIters=1000):
    print("Welcome to optimized_dp \n")
    # Initialize the HCL environment
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    ########################################## INITIALIZE ##########################################

    # Convert the python array to hcl type array
    V_opt = hcl.asarray(np.zeros(grid.pts_each_dim))
    
    gamma = hcl.asarray(gamma)
    actions = hcl.asarray(action_space)

    # Extract bounds from the grid object
    bounds = np.vstack((grid.min, grid.max)).T
    print(f"bounds is {bounds}")
    bounds = hcl.asarray(bounds)
    ptsEachDim = hcl.asarray(grid.pts_each_dim)
    
    f = _VALUE_ITERATIONS[bounds.shape[0]](MDP_obj, grid.pts_each_dim, bounds, action_space)

    # Build the graph and use the executable
    # Now use the executable
    t_s = time.time()
    iter = 0
    while iter < maxIters:
        print("Currently at iteration {}".format(iter))
        oldV = V_opt.asnumpy()
        f(V_opt, actions, gamma, bounds, ptsEachDim)
        diff = np.max(np.abs(oldV - V_opt.asnumpy()))
        print("Difference wrt to previous array {}".format(diff))
        if diff < epsilon:
            break
        iter += 1
    
    t_e = time.time()
    V = V_opt.asnumpy()
    print(f"Finished in {iter} iterations")
    print("Took        ", t_e - t_s, " seconds")
    
    return V

def HJSolver(dynamics_obj, grid, multiple_value, tau, compMethod,
             saveAllTimeSteps=False,
             accuracy="medium", untilConvergent=False, epsilon=2e-3,
             computeTimeToReach=False, solve_forward=False):
    
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

    # check accuracy doc string
    if accuracy not in ["low", "medium"]:
        raise ValueError("Accuracy must be either 'low' or 'medium'")
    
    if accuracy == "low":
        print("Using low accuracy, 1st order ENO scheme + 1st order TVD RK integration")
    elif accuracy == "medium":
        print("Using medium accuracy, 2nd order ENO scheme + 2nd order TVD RK integration")

    # Tensors input to our computation graph
    V_t = hcl.asarray(init_value)
    Hamiltonian = hcl.asarray(np.zeros(tuple(grid.pts_each_dim), dtype=np.float32))
    delta_t = hcl.asarray(np.zeros(1, dtype=np.float32))

    if computeTimeToReach:
        # Indirect TTR computation
        time_to_reach = np.ones(tuple(grid.pts_each_dim), dtype=np.float32) * 10000  # when attacker bound to win
        time_to_reach[init_value <= 0] = 0.
    
    # Check which target set or initial value set
    if compMethod["TargetSetMode"] != "minVWithVTarget" and compMethod["TargetSetMode"] != "maxVWithVTarget":
        l0 = hcl.asarray(init_value)
    else:
        l0 = hcl.asarray(target)

    # For debugging purposes
    #probe = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))

    # Array for each state values, in hcl array type
    list_xs = _state_arrays(grid)

    # Get executable, obstacle check intial value function
    # Only graph_3D supports forward reachability
    graph_kwargs = {"solve_forward": solve_forward} if grid.dims == 3 else {}
    solve_pde = _GRAPHS[grid.dims](dynamics_obj, grid, compMethod["TargetSetMode"], accuracy, **graph_kwargs)

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
            solve_pde(Hamiltonian, V_t, *list_xs, delta_t, t, l0)

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
                solve_pde(Hamiltonian, V_tp1, *list_xs, delta_t, t1, l0)
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
            
            if computeTimeToReach:
                # Update the time to reach
                update_idx = np.logical_and(V_tp1 <= 0, prev_arr > 0)
                time_to_reach[update_idx] = tNow
            
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
    if saveAllTimeSteps is True:
        valfuncs[..., 0] = V_t.asnumpy()
        if computeTimeToReach:
            return valfuncs, time_to_reach
        return valfuncs

    if computeTimeToReach:
        return V_t.asnumpy(), time_to_reach
    return V_t.asnumpy()

def TTRSolver(dynamics_obj, grid, multiple_value, epsilon, verbose=True):
    print("Welcome to optimized_dp TTRSolver \n")
    ################# INITIALIZE DATA TO BE INPUT INTO EXECUTABLE ##########################

    print("Initializing\n")
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    # Convert initial distance value function to initial time-to-reach value function
    if type(multiple_value) == list:
        # We have both goal and obstacle set
        target = multiple_value[0] # Target set
        obstacle = multiple_value[1] # Obstacle set
    else:
        target = multiple_value
        obstacle = None
        
    if obstacle is None:
        print("No obstacles set !")
        obstacle = np.full(target.shape, -1)
    else: 
        print("Obstacles set exists !")
        obstacle_dim = obstacle.ndim
        assert obstacle_dim == grid.dims, "We only support TTR with time-invariant obstacles now."
        obstacle = np.where(obstacle <= 0, 1000, 0)
    
    # Initial value function
    init_value = target
    init_value[init_value < 0] = 0
    init_value[init_value > 0] = 1000
    # Convert target and obstacle to heterocl syntax
    constraint = hcl.asarray(obstacle)
    V_0 = hcl.asarray(init_value)
    
    prev_val = np.zeros(init_value.shape)

    # States vectors in hcl array type
    list_xs = _state_arrays(grid)
    # TTR_1D does not take a constraint
    ttr_inputs = [V_0, *list_xs] if grid.dims == 1 else [V_0, constraint, *list_xs]

    # Get executable
    solve_TTR = _TTR_SOLVERS[grid.dims](dynamics_obj, grid)
    print("Got Executable\n")

    # Print out code for different backend
    # print(solve_pde)

    ################ USE THE EXECUTABLE ############
    error = 10000
    count = 0
    start = time.time()
    while error > epsilon:
        if verbose:
            print("Iteration: {} Error: {}".format(count, error))
        count += 1
        solve_TTR(*ttr_inputs)

        error = np.max(np.abs(prev_val - V_0.asnumpy()))
        prev_val = V_0.asnumpy()
                     
    # print("Total TTR computation time (s): {:.5f}".format(time.time() - start))
    # print("Finished solving\n")

    ##################### PLOTTING #####################
    return V_0.asnumpy()

def computeSpatDerivArray(grid, V, deriv_dim, accuracy="low"):
    # Return a tensor same size as V that contains spatial derivatives at every state in V
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    if accuracy not in ["low", "medium"]:
        raise ValueError("Accuracy must be either 'low' or 'medium'")
        
    # Need to make sure that value array has the same size as grid
    assert list(V.shape) == list(grid.pts_each_dim)

    V_0 = hcl.asarray(V)
    spatial_deriv = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))

    # Get executable, obstacle check intial value function
    compute_SpatDeriv = _GRAPHS[grid.dims](None, grid, "None", accuracy,
                                           generate_SpatDeriv=True, deriv_dim=deriv_dim)

    compute_SpatDeriv(V_0, spatial_deriv)
    return spatial_deriv.asnumpy()

