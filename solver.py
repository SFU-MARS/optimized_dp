import heterocl as hcl
import numpy as np
import time
import plotly.graph_objects as go

from CustomGraphFunctions import *
from user_definer import *
from argparse import ArgumentParser

import scipy.io as sio

import math


def main():
    ################### PARSING ARGUMENTS FROM USERS #####################

    parser = ArgumentParser()
    parser.add_argument("-p", "--plot", default=False, type=bool)
    # Print out LLVM option only
    parser.add_argument("-l", "--llvm", default=False, type=bool)
    args = parser.parse_args()

    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    ################## DATA SHAPE PREPARATION FOR GRAPH FORMATION  ####################
    V_f = hcl.placeholder(tuple(g.pts_each_dim), name="V_f", dtype = hcl.Float())
    V_init = hcl.placeholder(tuple(g.pts_each_dim), name="V_init", dtype=hcl.Float())
    l0 = hcl.placeholder(tuple(g.pts_each_dim), name="l0", dtype=hcl.Float())
    #x = hcl.placeholder((6, g.pts_each_dim[0]), name="x", dtype=hcl.Float())
    t = hcl.placeholder((2,), name="t", dtype=hcl.Float())

    # Deriv diff tensor
    deriv_diff1 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff1")
    deriv_diff2 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff2")
    deriv_diff3 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff3")
    deriv_diff4 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff4")
    deriv_diff5 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff5")
    deriv_diff6 = hcl.placeholder((tuple(g.pts_each_dim)), name="deriv_diff6")

    # Positions vector
    x1 = hcl.placeholder((g.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((g.pts_each_dim[1],), name="x2", dtype=hcl.Float())
    x3 = hcl.placeholder((g.pts_each_dim[2],), name="x3", dtype=hcl.Float())
    x4 = hcl.placeholder((g.pts_each_dim[3],), name="x4", dtype=hcl.Float())
    x5 = hcl.placeholder((g.pts_each_dim[4],), name="x5", dtype=hcl.Float())
    x6 = hcl.placeholder((g.pts_each_dim[5],), name="x6", dtype=hcl.Float())

    # Obstacle placeholder
    #obstacle = hcl.placeholder((tuple(g.pts_each_dim)), name="obstacle")
    
    ##################### CREATE SCHEDULE##############

    # Create schedule
    s = hcl.create_schedule(
        [V_f, V_init, deriv_diff1, deriv_diff2, deriv_diff3, deriv_diff4, deriv_diff5, deriv_diff6, x1, x2, x3, x4, x5, x6, t, l0], graph_6D)

    # Inspect the LLVM code
    print(hcl.lower(s))



    ################# INITIALIZE DATA TO BE INPUT INTO GRAPH ##########################

    print("Initializing\n")

    V_0 = hcl.asarray(my_shape)
    V_1 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    l0  = hcl.asarray(my_shape)
    #obstacle = hcl.asarray(cstraint_values)

    list_x1 = np.reshape(g.vs[0], g.pts_each_dim[0])
    list_x2 = np.reshape(g.vs[1], g.pts_each_dim[1])
    list_x3 = np.reshape(g.vs[2], g.pts_each_dim[2])
    list_x4 = np.reshape(g.vs[3], g.pts_each_dim[3])
    list_x5 = np.reshape(g.vs[4], g.pts_each_dim[4])
    list_x6 = np.reshape(g.vs[5], g.pts_each_dim[5])

    # Convert to hcl array type
    list_x1 = hcl.asarray(list_x1)
    list_x2 = hcl.asarray(list_x2)
    list_x3 = hcl.asarray(list_x3)
    list_x4 = hcl.asarray(list_x4)
    list_x5 = hcl.asarray(list_x5)
    list_x6 = hcl.asarray(list_x6)

    # Initialize deriv diff tensor
    deriv_diff1 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    deriv_diff2 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    deriv_diff3 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    deriv_diff4 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    deriv_diff5 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    deriv_diff6 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))


    ##################### CODE OPTIMIZATION HERE ###########################
    print("Optimizing\n")

    # Accessing the hamiltonian stage
    s_H = graph_6D.Hamiltonian
    s_D = graph_6D.Dissipation

    #
    s[s_H].parallel(s_H.i)
    s[s_D].parallel(s_D.i)

    # Inspect IR
    #if args.llvm:
    #    print(hcl.lower(s))

    ################ GET EXECUTABLE AND USE THE EXECUTABLE ############
    print("Running\n")

    # Get executable
    solve_pde = hcl.build(s)

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

             # Run the execution and pass input into graph
             solve_pde(V_1, V_0, deriv_diff1, deriv_diff2, deriv_diff3, deriv_diff4, deriv_diff5, deriv_diff6,
                       list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, t_minh, l0)

             tNow = np.asscalar((t_minh.asnumpy())[0])

             if lookback_time != 0: # Exclude first time of the computation
                 execution_time += time.time() - start

             # Some information printing
             print(t_minh)
             print("Computational time to integrate (s): {:.5f}".format(time.time() - start))
             # Saving data into disk
             if tNow >= tau[i] - 1e-4:
                 print("Saving files\n")
                 sio.savemat('/local-scratch/Humannoid/humannoid_v_{:d}.mat'.format(i), {'V_array': V_1.asnumpy()})
    
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
