import heterocl as hcl
import numpy as np
import time
import plotly.graph_objects as go
from GridProcessing import Grid
from ShapesFunctions import *
from CustomGraphFunctions import *
from DubinsCar import *

import math

""" USER INTERFACES
- Define grid

- Generate initial values for grid using shape functions

- Time length for computations

- Run
"""

# Create a grid
g = grid(np.array([-5.0, -5.0, -math.pi]), np.array([5.0, 5.0, math.pi]), 3 ,np.array([100,100,100]), 2)
# Use the grid to initialize initial value function
shape = CyclinderShape(g, 3, np.zeros(3), 1)
# Define my car
myCar = DubinsCar(x=np.array([0,0,0]), wMax=1, speed=1, dMax=np.array([0,0,0]), uMode="min", dMode="max")

# Look-back lenght and time step
lookback_length = 1.00
t_step = 0.05




def HJ_PDE_solver(V_new, V_init, thetas ,t):

    # These variables are used to dissipation calculation
    max_alpha1 = hcl.scalar(-1e9, "max_alpha1")
    max_alpha2 = hcl.scalar(-1e9, "max_alpha2")
    max_alpha3 = hcl.scalar(-1e9, "max_alpha3")

    # Calculate spatial derivative
    def spa_derivX(i, j, k):
        left_deriv = hcl.scalar(0, "left_deriv")
        right_deriv = hcl.scalar(0, "right_deriv")
        with hcl.if_(i == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V_init[i, j, k] + my_abs(V_init[i + 1, j, k] - V_init[i, j, k]) * my_sign(
                V_init[i, j, k])
            left_deriv[0] = (V_init[i, j, k] - left_boundary[0]) / g.dx[0]
            right_deriv[0] = (V_init[i + 1, j, k] - V_init[i, j, k]) / g.dx[0]
        with hcl.elif_(i == V_init.shape[0] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V_init[i, j, k] + my_abs(V_init[i, j, k] - V_init[i - 1, j, k]) * my_sign(
                V_init[i, j, k])
            left_deriv[0] = (V_init[i, j, k] - V_init[i - 1, j, k]) / g.dx[0]
            right_deriv[0] = (right_boundary[0] - V_init[i, j, k]) / g.dx[0]
        with hcl.elif_(i != 0 and i != V_init.shape[0] - 1):
            left_deriv[0] = (V_init[i, j, k] - V_init[i - 1, j, k]) / g.dx[0]
            right_deriv[0] = (V_init[i + 1, j, k] - V_init[i, j, k]) / g.dx[0]
        return left_deriv[0], right_deriv[0]

    def spa_derivY(i, j, k):
        left_deriv = hcl.scalar(0, "left_deriv")
        right_deriv = hcl.scalar(0, "right_deriv")
        with hcl.if_(j == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            left_boundary[0] = V_init[i, j, k] + my_abs(V_init[i, j + 1, k] - V_init[i, j, k]) * my_sign(
                V_init[i, j, k])
            left_deriv[0] = (V_init[i, j, k] - left_boundary[0]) / g.dx[1]
            right_deriv[0] = (V_init[i, j + 1, k] - V_init[i, j, k]) / g.dx[1]
        with hcl.elif_(j == V_init.shape[1] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V_init[i, j, k] + my_abs(V_init[i, j, k] - V_init[i, j - 1, k]) * my_sign(
                V_init[i, j, k])
            left_deriv[0] = (V_init[i, j, k] - V_init[i, j - 1, k]) / g.dx[1]
            right_deriv[0] = (right_boundary[0] - V_init[i, j, k]) / g.dx[1]
        with hcl.elif_(j != 0 and j != V_init.shape[1] - 1):
            left_deriv[0] = (V_init[i, j, k] - V_init[i, j - 1, k]) / g.dx[1]
            right_deriv[0] = (V_init[i, j + 1, k] - V_init[i, j, k]) / g.dx[1]
        return left_deriv[0], right_deriv[0]

    def spa_derivT(i, j, k):
        left_deriv = hcl.scalar(0, "left_deriv")
        right_deriv = hcl.scalar(0, "right_deriv")
        with hcl.if_(k == 0):
            left_boundary = hcl.scalar(0, "left_boundary")
            # left_boundary[0] = V_init[i,j,50]
            left_boundary[0] = V_init[i, j, V_init.shape[2] - 1]
            left_deriv[0] = (V_init[i, j, k] - left_boundary[0]) / g.dx[2]
            right_deriv[0] = (V_init[i, j, k + 1] - V_init[i, j, k]) / g.dx[2]
        with hcl.elif_(k == V_init.shape[2] - 1):
            right_boundary = hcl.scalar(0, "right_boundary")
            right_boundary[0] = V_init[i, j, 0]
            left_deriv[0] = (V_init[i, j, k] - V_init[i, j, k - 1]) / g.dx[2]
            right_deriv[0] = (right_boundary[0] - V_init[i, j, k]) / g.dx[2]
        with hcl.elif_(k != 0 and k != V_init.shape[2] - 1):
            left_deriv[0] = (V_init[i, j, k] - V_init[i, j, k - 1]) / g.dx[2]
            right_deriv[0] = (V_init[i, j, k + 1] - V_init[i, j, k]) / g.dx[2]
        return left_deriv[0], right_deriv[0]

    def step_bound(): # Function to calculate time step
        stepBoundInv = hcl.scalar(0, "stepBoundInv")
        stepBound    = hcl.scalar(0, "stepBound")
        stepBoundInv[0] = max_alpha1[0]/g.dx[0] + max_alpha2[0]/g.dx[1] + max_alpha3[0]/g.dx[2]

        stepBound[0] = 0.8/stepBoundInv[0]
        with hcl.if_(stepBound > t_step):
            stepBound[0] = t_step
        time = stepBound[0]
        return time

    # Calculate Hamiltonian for every grid point in V_init
    with hcl.Stage("Hamiltonian"):
        with hcl.for_(0, V_init.shape[0], name="k") as k: # Plus 1 as for loop count stops at V_init.shape[0]
            with hcl.for_(0, V_init.shape[1], name="j") as j:
                with hcl.for_(0, V_init.shape[2], name="i") as i:
                    # Variables to calculate dV_dx
                    dV_dx_L = hcl.scalar(0, "dV_dx_L")
                    dV_dx_R = hcl.scalar(0, "dV_dx_R")
                    dV_dx = hcl.scalar(0, "dV_dx")
                    # Variables to calculate dV_dy
                    dV_dy_L = hcl.scalar(0, "dV_dy_L")
                    dV_dy_R = hcl.scalar(0, "dV_dy_R")
                    dV_dy = hcl.scalar(0, "dV_dy")
                    # Variables to calculate dV_dtheta
                    dV_dT_L = hcl.scalar(0, "dV_dT_L")
                    dV_dT_R = hcl.scalar(0, "dV_dT_R")
                    dV_dT = hcl.scalar(0, "dV_dT")
                    # Variables to keep track of dynamics
                    #dx_dt = hcl.scalar(0, "dx_dt")
                    #dy_dt = hcl.scalar(0, "dy_dt")
                    #dtheta_dt = hcl.scalar(0, "dtheta_dt")

                    # No tensor slice operation
                    dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
                    dV_dy_L[0], dV_dy_R[0] = spa_derivY(i, j, k)
                    dV_dT_L[0], dV_dT_R[0] = spa_derivT(i, j, k)

                    # Calculate average gradient
                    dV_dx[0] = (dV_dx_L + dV_dx_R) / 2
                    dV_dy[0] = (dV_dy_L + dV_dy_R) / 2
                    dV_dT[0] = (dV_dT_L + dV_dT_R) / 2

                    # Use method of DubinsCar to solve optimal control instead
                    uOpt = myCar.opt_ctrl((dV_dx[0], dV_dy[0], dV_dT[0]))

                    # Calculate dynamical rates of changes
                    dx_dt, dy_dt, dtheta_dt = myCar.dynamics(thetas[k], uOpt)

                    # Calculate Hamiltonian terms:
                    V_new[i, j, k] =  -(dx_dt * dV_dx[0] + dy_dt * dV_dy[0] + dtheta_dt * dV_dT[0])

                    # Calculate dissipation step
                    dx_dt = my_abs(dx_dt)
                    dy_dt = my_abs(dy_dt)
                    dtheta_dt = my_abs(dtheta_dt)
                    diss = hcl.scalar(0, "diss")
                    diss[0] = 0.5*((dV_dx_R[0] - dV_dx_L[0])*dx_dt + (dV_dy_R[0] - dV_dy_L[0])*dy_dt + (dV_dT_R[0] - dV_dT_L[0])* dtheta_dt)
                    V_new[i, j, k] = -(V_new[i, j, k] - diss[0])

                    # Calculate alphas
                    with hcl.if_(dx_dt > max_alpha1):
                        max_alpha1[0] = dx_dt
                    with hcl.if_(dy_dt > max_alpha2):
                        max_alpha2[0] = dy_dt
                    with hcl.if_(dtheta_dt > max_alpha3):
                        max_alpha3[0] = dtheta_dt

    # Determine time step
    hcl.update(t, lambda x: step_bound())
    # Integrate
    result = hcl.update(V_new, lambda i,j,k: V_init[i,j,k] + V_new[i,j,k] * t[0])
    # Copy V_new to V_init
    hcl.update(V_init, lambda i,j,k: V_new[i,j,k] )
    return result

def main():
    hcl.init()
    hcl.config.init_dtype = hcl.Float()
    V_f = hcl.placeholder(tuple(g.pts_each_dim), name="V_f", dtype = hcl.Float())
    V_init = hcl.placeholder(tuple(g.pts_each_dim), name="V_init", dtype=hcl.Float())
    thetas = hcl.placeholder((g.pts_each_dim[2],), name="thetas", dtype=hcl.Float())
    t = hcl.placeholder((1,), name="t", dtype=hcl.Float())

    # Create schedule
    s = hcl.create_schedule([V_f, V_init, thetas,t], HJ_PDE_solver)

    # Here comes the optimization

    # Accessing the hamiltonian stage
    s_H = HJ_PDE_solver.Hamiltonian
    # Split the loops
    k_out, k_in = s[s_H].split(s_H.k, 10) # These numbers are experimental, changable
    j_out, j_in = s[s_H].split(s_H.j, 10)
    i_out, i_in = s[s_H].split(s_H.i, 10)

    # Reorder the loops
    s[s_H].reorder(j_out, k_in)
    s[s_H].reorder(i_out, k_in)
    s[s_H].reorder(k_in, j_in)

    # FPGA Back end - parallel specs
    s[s_H].pipeline(k_in)
    s[s_H].unroll(i_out, 5)

    # If CPU option
    s[s_H].parallel(k_out)

    # Inspect IR
    #print(hcl.lower(s))

    # Build the code
    solve_pde = hcl.build(s)

    #print(f)

    # Prepare numpy array for graph computation
    V_0 = hcl.asarray(shape)
    V_1=  hcl.asarray(np.zeros(tuple(g.pts_each_dim)))


    t_minh = hcl.asarray(np.zeros(1))

    # List thetas
    list_theta = np.reshape(g.vs[2], g.pts_each_dim[2])
    list_theta = hcl.asarray(list_theta)


    # Variables used for timing
    execution_time = 0
    lookback_time = 0

    print("I'm here\n")
    # Test the executable from heteroCL:
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


    #V = V_1.asnumpy()
    #V = np.swapaxes(V, 0,2)
    #V = np.swapaxes(V, 1,2)
    #probe = probe.asnumpy()
    #probe = np.swapaxes(probe, 0, 2)
    #probe = np.swapaxes(probe, 1, 2)
    #print(V)
    #V_1 = V_1.asnumpy()

    # Time info printing
    print("Total kernel time (s): {:.5f}".format(execution_time))
    print("Finished solving\n")

    # Plotting
    print("Plotting beautiful plots. Please wait\n")
    fig = go.Figure(data=go.Isosurface(
        x=g.mg_X.flatten(),
        y=g.mg_Y.flatten(),
        z=g.mg_T.flatten(),
        value=V_1.asnumpy().flatten(),
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
