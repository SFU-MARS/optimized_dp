import heterocl as hcl
import numpy as np
import time

import computeGraphs
""" 4D DUBINS CAR DYNAMICS IMPLEMENTATION 
 x_dot = v * cos(theta)
 y_dot = v * sin(theta)
 v_dot = a
 theta_dot = w
 """
class DubinsCar4D:
    def __init__(self, x=[0,0,0,0], uMin = [-1,-1], uMax = [1,1], dMin = [-0.25,-0.25], \
                 dMax=[0.25,0.25], uMode="min", dMode="max"):
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # System dynamics
        # x_dot     = v * cos(theta) + d_1
        # y_dot     = v * sin(theta) + d_2
        # v_dot     = a
        # theta_dot = w

        # Graph takes in 4 possible inputs, by default, for now
        opt_a = hcl.scalar(self.uMax[0], "opt_a")
        opt_w = hcl.scalar(self.uMax[1], "opt_w")
        # Just create and pass back, even though they're not used
        in3   = hcl.scalar(0, "in3")
        in4   = hcl.scalar(0, "in4")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[2] > 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(spat_deriv[3] > 0):
                opt_w[0] = self.uMin[1]
        else:
            with hcl.if_(spat_deriv[2] < 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(spat_deriv[3] < 0):
                opt_w[0] = self.uMin[1]
        # return 3, 4 even if you don't use them
        return (opt_a[0] ,opt_w[0], in3[0], in4[0])

    def optDstb(self, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        # Just create and pass back, even though they're not used
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        #with hcl.if_(self.dMode == "max"):
        if self.dMode == "max":
            with hcl.if_(spat_deriv[0] > 0):
                d1[0] = self.dMax[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMin[0]
            with hcl.if_(spat_deriv[1] > 0):
                d2[0] = self.dMax[1]
            with hcl.elif_(spat_deriv[1] < 0):
                d2[0] = self.dMin[1]

        else:
            with hcl.if_(spat_deriv[0] > 0):
                d1[0] = self.dMin[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMax[0]
            with hcl.if_(spat_deriv[1] > 0):
                d2[0] = self.dMin[1]
            with hcl.elif_(spat_deriv[1] < 0):
                d2[0] = self.dMax[1]


        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        v_dot = hcl.scalar(0, "v_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = state[2] * hcl.cos(state[3]) + dOpt[0]
        y_dot[0] = state[2] * hcl.sin(state[3]) + dOpt[1]
        v_dot[0] = uOpt[0]
        theta_dot[0] = uOpt[1]

        return (x_dot[0], y_dot[0], v_dot[0] ,theta_dot[0])
