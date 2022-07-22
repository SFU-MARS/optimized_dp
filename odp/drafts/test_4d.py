import heterocl as hcl
import numpy as np
import time
import plotly.graph_objects as go
from GridProcessing import Grid
from ShapesFunctions import *
from CustomGraphFunctions import *
from InvertPendulum import *
from argparse import ArgumentParser


""" USER INTERFACES
- Define grid

- Generate initial values for grid using shape functions

- Time length for computations

- Run
"""

# What value to use for grid
g = grid(np.array([-3.0, -1.0, -3.0, -1.0]), np.array([3.0, 1.0, 3.0, 1.0]), 4 ,np.array([60,60,60, 60])) # Leave out periodic field
# Use the grid to initialize initial value function
my_shape = Rectangle4D(g, [[-0.05, 0.05], [-0.01, 0.01], [0.99, 1.01], [-0.01, 0.01]])
# Define my object
myPendulum = inverted_pendulum_2d()

# Look-back lenght and time step
lookback_length = 1.00
t_step = 0.05




def graph_4D(V_new, V_init, deriv_diff1, deriv_diff2, deriv_diff3, deriv_diff4, x, t):
    # Maximum derivative for each dim
    max_deriv1 = hcl.scalar(-1e9, "max_deriv1")
    max_deriv2 = hcl.scalar(-1e9, "max_deriv2")
    max_deriv3 = hcl.scalar(-1e9, "max_deriv3")
    max_deriv4 = hcl.scalar(-1e9, "max_deriv4")

    # Min derivative for each dim
    min_deriv1 = hcl.scalar(1e9, "min_deriv1")
    min_deriv2 = hcl.scalar(1e9, "min_deriv2")
    min_deriv3 = hcl.scalar(1e9, "min_deriv3")
    min_deriv4 = hcl.scalar(1e9, "min_deriv4")

    # These variables are used to dissipation calculation
    max_alpha1 = hcl.scalar(-1e9, "max_alpha1")
    max_alpha2 = hcl.scalar(-1e9, "max_alpha2")
    max_alpha3 = hcl.scalar(-1e9, "max_alpha3")
    max_alpha4 = hcl.scalar(-1e9, "max_alpha4")

    def step_bound(): # Function to calculate time step
        stepBoundInv = hcl.scalar(0, "stepBoundInv")
        stepBound    = hcl.scalar(0, "stepBound")
        stepBoundInv[0] = max_alpha1[0]/g.dx[0] + max_alpha2[0]/g.dx[1] + max_alpha3[0]/g.dx[2] + max_alpha4[0]/g.dx[3]

        stepBound[0] = 0.8/stepBoundInv[0]
        with hcl.if_(stepBound > t_step):
            stepBound[0] = t_step
        time = stepBound[0]
        return time

    # Calculate Hamiltonian for every grid point in V_init
    with hcl.Stage("Hamiltonian"):
        with hcl.for_(0, V_init.shape[0], name="i") as i:
            with hcl.for_(0, V_init.shape[1], name="j") as j:
                with hcl.for_(0, V_init.shape[2], name="k") as k:
                    with hcl.for_(0, V_init.shape[3], name="l") as l:
                        #Variables to calculate dV_dx
                        dV_dx1_L = hcl.scalar(0, "dV_dx1_L")
                        dV_dx1_R = hcl.scalar(0, "dV_dx1_R")
                        dV_dx1 = hcl.scalar(0, "dV_dx1")
                        dV_dx2_L = hcl.scalar(0, "dV_dx2_L")
                        dV_dx2_R = hcl.scalar(0, "dV_dx2_R")
                        dV_dx2 = hcl.scalar(0, "dV_dx2")
                        dV_dx3_L = hcl.scalar(0, "dV_dx3_L")
                        dV_dx3_R = hcl.scalar(0, "dV_dx3_R")
                        dV_dx3 = hcl.scalar(0, "dV_dx3")
                        dV_dx4_L = hcl.scalar(0, "dV_dx4_L")
                        dV_dx4_R = hcl.scalar(0, "dV_dx4_R")
                        dV_dx4 = hcl.scalar(0, "dV_dx4")

                        # No tensor slice operation
                        #dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
                        dV_dx1_L[0], dV_dx1_R[0] = spa_derivX1_4d(i, j, k, l, V_init, g)
                        dV_dx2_L[0], dV_dx2_R[0] = spa_derivX2_4d(i, j, k, l, V_init, g)
                        dV_dx3_L[0], dV_dx3_R[0] = spa_derivX3_4d(i, j, k, l, V_init, g)
                        dV_dx4_L[0], dV_dx4_R[0] = spa_derivX4_4d(i, j, k, l, V_init, g)

                        # Saves spatial derivative diff into tables
                        deriv_diff1[i, j, k, l] = dV_dx1_R[0] - dV_dx1_L[0]
                        deriv_diff2[i, j, k, l] = dV_dx2_R[0] - dV_dx2_L[0]
                        deriv_diff3[i, j, k, l] = dV_dx3_R[0] - dV_dx3_L[0]
                        deriv_diff4[i, j, k, l] = dV_dx4_R[0] - dV_dx4_L[0]

                        # Calculate average gradient
                        dV_dx1[0] = (dV_dx1_L + dV_dx1_R) / 2
                        dV_dx2[0] = (dV_dx2_L + dV_dx2_R) / 2
                        dV_dx3[0] = (dV_dx3_L + dV_dx3_R) / 2
                        dV_dx4[0] = (dV_dx4_L + dV_dx4_R) / 2


                        # Find optimal control
                        uOpt = myPendulum.opt_ctrl(x, (dV_dx1[0], dV_dx2[0], dV_dx3[0], dV_dx4[0]))

                        # Find rates of changes based on dynamics equation
                        dx1_dt, dx2_dt, dx3_dt, dx4_dt = myPendulum.dynamics(x, uOpt)

                        # Calculate Hamiltonian terms:
                        V_new[i, j, k, l] =  - myPendulum.Hamiltonian((dx1_dt, dx2_dt, dx3_dt, dx4_dt), (dV_dx1, dV_dx2, dV_dx3, dV_dx4))

                        # Get derivMin
                        with hcl._if(dV_dx1_L[0] < min_deriv1):
                            min_deriv1[0] = dV_dx1_L[0]
                        with hcl._if(dV_dx1_R[0] < min_deriv1):
                            min_deriv1[0] = dV_dx1_R[0]

                        with hcl._if(dV_dx2_L[0] < min_deriv2):
                            min_deriv2[0] = dV_dx2_L[0]
                        with hcl._if(dV_dx2_R[0] < min_deriv2):
                            min_deriv2[0] = dV_dx2_R[0]

                        with hcl._if(dV_dx3_L[0] < min_deriv3):
                            min_deriv3[0] = dV_dx3_L[0]
                        with hcl._if(dV_dx3_R[0] < min_deriv3):
                            min_deriv3[0] = dV_dx3_R[0]

                        with hcl._if(dV_dx4_L[0] < min_deriv4):
                            min_deriv4[0] = dV_dx4_L[0]
                        with hcl._if(dV_dx4_R[0] < min_deriv4):
                            min_deriv4[0] = dV_dx4_R[0]

                        # Get derivMax
                        with hcl._if(dV_dx1_L[0] > max_deriv1):
                            max_deriv1[0] = dV_dx1_L[0]
                        with hcl._if(dV_dx1_R[0] > max_deriv1):
                            max_deriv1[0] = dV_dx1_R[0]

                        with hcl._if(dV_dx2_L[0] > max_deriv2):
                            max_deriv2[0] = dV_dx2_L[0]
                        with hcl._if(dV_dx2_R[0] > max_deriv2):
                            max_deriv2[0] = dV_dx2_R[0]

                        with hcl._if(dV_dx3_L[0] > max_deriv3):
                            max_deriv3[0] = dV_dx3_L[0]
                        with hcl._if(dV_dx3_R[0] > max_deriv3):
                            max_deriv3[0] = dV_dx3_R[0]

                        with hcl._if(dV_dx4_L[0] > max_deriv4):
                            max_deriv4[0] = dV_dx4_L[0]
                        with hcl._if(dV_dx4_R[0] > max_deriv4):
                            max_deriv4[0] = dV_dx4_R[0]

    # Calculate dissipation amount
    with hcl.Stage("Dissipation"):
        # Find LOWER BOUND optimal control
        uOptL = myPendulum.opt_ctrl(x, (min_deriv1[0], min_deriv2[0], min_deriv3[0], min_deriv4[0]))
        # Find UPPER BOUND optimal control
        uOptU = myPendulum.opt_ctrl(x, (max_deriv1[0], max_deriv2[0], max_deriv3[0], max_deriv4[0]))
        with hcl.for_(0, V_init.shape[0], name="i") as i:
            with hcl.for_(0, V_init.shape[1], name="j") as j:
                with hcl.for_(0, V_init.shape[2], name="k") as k:
                    with hcl.for_(0, V_init.shape[3], name="l") as l:
                        dx_L1, dx_L2, dx_L3, dx_L4 = myPendulum.dynamics(x, uOptL)
                        dx_U1, dx_U2, dx_U3, dx_U4 = myPendulum.dynamics(x, uOptU)

                        alpha1 = my_max(my_abs(dx_L1), my_abs(dx_U1))
                        alpha2 = my_max(my_abs(dx_L1), my_abs(dx_U1))
                        alpha3 = my_max(my_abs(dx_L3), my_abs(dx_U3))
                        alpha4 = my_max(my_abs(dx_L4), my_abs(dx_U4))

                        diss = hcl.scalar(0, "diss")
                        diss[0] = 0.5*(deriv_diff1[i, j, k, l]*alpha1 + deriv_diff2[i, j, k, l]*alpha2 + deriv_diff3[i, j, k, l]* alpha3 + deriv_diff4[i, j, k, l]* alpha4)

                        # Finally
                        V_new[i, j, k, l] = -(V_new[i, j, k, l] - diss[0])
                        # Get maximum alphas in each dimension

                        # Calculate alphas
                        with hcl.if_(alpha1 > max_alpha1):
                            max_alpha1[0] = alpha1
                        with hcl.if_(alpha2 > max_alpha2):
                            max_alpha2[0] = alpha2
                        with hcl.if_(alpha3 > max_alpha3):
                            max_alpha3[0] = alpha3
                        with hcl.if_(alpha4 > max_alpha4):
                            max_alpha4[0] = alpha4


    # Determine time step
    hcl.update(t, lambda x: step_bound())
    # Integrate
    result = hcl.update(V_new, lambda i, j, k: V_init[i,j,k] + V_new[i,j,k] * t[0])
    # Copy V_new to V_init
    hcl.update(V_init, lambda i,j,k: V_new[i,j,k] )
    return result

def main():
    ################### PARSING ARGUMENTS FROM USERS #####################

    parser = ArgumentParser()
    parser.add_argument("-p", "--plot", default=False, type=bool)
    args = parser.parse_args()

    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    ################## DATA SHAPE PREPARATION FOR GRAPH FORMATION  ####################
    V_f = hcl.placeholder(tuple(g.pts_each_dim), name="V_f", dtype = hcl.Float())
    V_init = hcl.placeholder(tuple(g.pts_each_dim), name="V_init", dtype=hcl.Float())
    x = hcl.placeholder((4, g.pts_each_dim[0]), name="x", dtype=hcl.Float())
    t = hcl.placeholder((1,), name="t", dtype=hcl.Float())

    # Deriv diff tensor
    deriv_diff1 = hcl.Tensor((tuple(g.pts_each_dim)), name="deriv_diff1")
    deriv_diff2 = hcl.Tensor((tuple(g.pts_each_dim)), name="deriv_diff2")
    deriv_diff3 = hcl.Tensor((tuple(g.pts_each_dim)), name="deriv_diff3")
    deriv_diff4 = hcl.Tensor((tuple(g.pts_each_dim)), name="deriv_diff4")
    
    ################# INITIALIZE DATA TO BE INPUT INTO GRAPH ##########################

    V_0 = hcl.asarray(my_shape)
    V_1=  hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    t_minh = hcl.asarray(np.zeros(1))

    x = np.zeros(4, g.pts_each_dim[0])
    for i in range(0,4):
        for j in range(0, g.pts_each_dim[i]):
            x[i,j] = g.xs[i][j]

    # Convert x to hcl array type
    x = hcl.asarray(x)

    ##################### CREATE SCHEDULE##############

    # Create schedule
    s = hcl.create_schedule([V_f, V_init, x, t], graph_4D)

    # Inspect the LLVM code
    print(hcl.lower(s))

    ##################### CODE OPTIMIZATION HERE ###########################

    # Accessing the hamiltonian stage
    graph_4D.Hamiltonian

    #


    ################ GET EXECUTABLE AND USE THE EXECUTABLE ############
    # Get executable
    solve_pde = hcl.build(s)


    # Variables used for timing
    execution_time = 0
    lookback_time = 0
    while lookback_time <= lookback_length:
        # Start timing
        start = time.time()

        # Printing some info
        #print("Look back time is (s): {:.5f}".format(lookback_time))

        # Run the execution and pass input into graph
        solve_pde(V_1, V_0, list_theta, t_minh)

        if lookback_time != 0: # Exclude first time of the computation
            execution_time += time.time() - start
        lookback_time += np.asscalar(t_minh.asnumpy())

        # Some information printing
        #print(t_minh)
        print("Computational time to integrate (s): {:.5f}".format(time.time() - start))


    V_1 = V_1.asnumpy()
    V_1 = np.swapaxes(V_1, 0,2)
    #V = np.swapaxes(V, 1,2)
    #probe = probe.asnumpy()
    #probe = np.swapaxes(probe, 0, 2)
    #probe = np.swapaxes(probe, 1, 2)
    #print(V)
    #V_1 = V_1.asnumpy()


    # Time info printing
    print("Total kernel time (s): {:.5f}".format(execution_time))
    print("Finished solving\n")

    ##################### PLOTTING #####################
    if args.plot:
        print("Plotting beautiful plots. Please wait\n")
        fig = go.Figure(data=go.Isosurface(
            x=g.mg_X.flatten(),
            y=g.mg_Y.flatten(),
            z=g.mg_T.flatten(),
            value=V_1.flatten(),
            colorscale='jet',
            isomin=0,
            surface_count=1,
            isomax=0,
            caps=dict(x_show=True, y_show=True)
        ))
        fig.show()
        print("Please check the plot on your browser.")


if __name__ == '__main__':
  main()
