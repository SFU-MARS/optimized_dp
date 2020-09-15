import heterocl as hcl
import numpy as np
import time
#import plotly.graph_objects as go
import os

from computeGraphs.CustomGraphFunctions import *
from Plots.plotting_utilities import *
from user_definer import *
from argparse import ArgumentParser
from computeGraphs.graph_4D import *
from computeGraphs.graph_5D import *
from computeGraphs.graph_6D import *
import scipy.io as sio

import matplotlib.pyplot as plt

import math

from prediction.clustering_v3 import ClusteringV3
from prediction.process_prediction_v3 import ProcessPredictionV3

def main():
    ################### PARSING ARGUMENTS FROM USERS #####################

    parser = ArgumentParser()
    parser.add_argument("-p", "--plot", default=True, type=bool)
    # Print out LLVM option only
    parser.add_argument("-l", "--llvm", default=False, type=bool)
    args = parser.parse_args()

    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    ################# Obtain and reset the action bound for each round of computation #################
    # Get the action bound for each mode
    # action_bound_mode = [mode_num, acc_min, acc_max, omega_min, omega_max]
    action_bound_mode = ClusteringV3().get_clustering()
    omega_bound, acc_bound = ProcessPredictionV3().omega_bound, ProcessPredictionV3().acc_bound

    # A bigger loop than the main loop
    # This loops over different driving mode and set different action bound
    for mode_num in range(-1, ClusteringV3().clustering_num):
        # Reset the my_car object and pass it into main loop
        # Disturbances: w_h, a_h

        # Mode -1, full range
        if mode_num == -1:
            my_car.dMin, my_car.dMax = \
                np.array([omega_bound[0], acc_bound[0]]), \
                np.array([omega_bound[1], acc_bound[1]])
        else:
            my_car.dMin, my_car.dMax = \
                np.array([action_bound_mode[mode_num][3], action_bound_mode[mode_num][1]]), \
                np.array([action_bound_mode[mode_num][4], action_bound_mode[mode_num][2]])
        print("reset to the driving mode", mode_num)
        print("In current mode, dmin is", my_car.dMin, "dmax are", my_car.dMax)

        ################# INITIALIZE DATA TO BE INPUT INTO EXECUTABLE ##########################

        print("Initializing\n")

        V_0 = hcl.asarray(my_shape)
        V_1 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
        l0  = hcl.asarray(my_shape)
        #probe = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
        #obstacle = hcl.asarray(cstraint_values)

        # Initialize uOpt_1, uOpt_2
        uOpt_1 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
        uOpt_2 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))

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
        print(solve_pde)

        ################ USE THE EXECUTABLE ############
        # Variables used for timing
        execution_time = 0
        lookback_time = 0

        # # Get the action bound for each mode
        # # action_bound_mode = [mode_num, acc_min, acc_max, omega_min, omega_max]
        # action_bound_mode = ClusteringV3().get_clustering()
        # omega_bound, acc_bound = ProcessPredictionV3().omega_bound, ProcessPredictionV3().acc_bound
        #
        # # A bigger loop than the main loop
        # # This loops over different driving mode and set different action bound
        # for mode_num in range(0, ClusteringV3().clustering_num):
        #     # Reset the my_car object and pass it into main loop
        #     # Disturbances: w_h, a_h
        #
        #     # Mode -1, full range
        #     if mode_num == -1:
        #         my_car.dMin, my_car.dMax = \
        #             np.array([ProcessPredictionV3().omega_bound[0], ProcessPredictionV3().acc_bound[0]]), \
        #             np.array([ProcessPredictionV3().omega_bound[1], ProcessPredictionV3().acc_bound[1]])
        #     else:
        #         my_car.dMin, my_car.dMax = \
        #             np.array([action_bound_mode[mode_num][3], action_bound_mode[mode_num][1]]), \
        #             np.array([action_bound_mode[mode_num][4], action_bound_mode[mode_num][2]])
        #         print("dmin, dmax are", my_car.dMin, my_car.dMax)


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
                    solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5 ,t_minh, l0, uOpt_1, uOpt_2)
                 if g.dims == 6:
                    solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, t_minh, l0)

                 # tNow = np.asscalar((t_minh.asnumpy())[0])
                 tNow = (t_minh.asnumpy())[0].item() # Above line is deprecated since NumPy v1.16

                 # Calculate computation time
                 execution_time += time.time() - start

                 # Some information printing
                 print(t_minh)
                 print("Computational time to integrate (s): {:.5f}".format(time.time() - start))

                 print("tNow is ", tNow)
                 # Saving reachable set and control data into disk
                 if tNow == 0.0029314488638192415 or tNow == 0.5 or tNow == 1 or tNow == 1.5 or tNow == 2 or tNow == 2.5 or tNow == 3:
                    file_dir = '/home/anjianl/Desktop/project/optimized_dp/data/brs/0910/mode' + str(mode_num)
                    if not os.path.exists(file_dir):
                        os.mkdir(file_dir)
                    file_brs_path = file_dir + '/reldyn5d_brs_mode' + str(mode_num) + '_t_%.2f.npy'
                    np.save(file_brs_path % tNow, V_1.asnumpy())
                    print("reachable set for mode", mode_num, "is saved!")

                    # Save control data
                    file_ctrl_beta_path = file_dir + '/reldyn5d_ctrl_beta_mode' + str(mode_num) + '_t_%.2f.npy'
                    file_ctrl_acc_path = file_dir + '/reldyn5d_ctrl_acc_mode' + str(mode_num) + '_t_%.2f.npy'
                    np.save(file_ctrl_beta_path % tNow, uOpt_1.asnumpy())
                    print("control beta for mode", mode_num, "is saved!")
                    np.save(file_ctrl_acc_path % tNow, uOpt_2.asnumpy())
                    print("control acc for mode", mode_num, "is saved!")

                    a = 1

        # Time info printing
        print("Total kernel time (s): {:.5f}".format(execution_time))
        print("Finished solving\n")

        # V1 is the final value array, fill in anything to use it



        ##################### PLOTTING #####################
        # if args.plot:
        #     plot_isosurface(g, V_1.asnumpy(), [0, 1, 3])


if __name__ == '__main__':

    main()
