import heterocl as hcl
import numpy as np
import time

""" 4D DUBINS CAR DYNAMICS IMPLEMENTATION 
 x_dot = v * cos(theta)
 y_dot = v * sin(theta)
 v_dot = a
 theta_dot = w
 """
class DubinsCar4D:
    def __init__(self, x=[0,0,0,0], uMin = [-1,-1], uMax = [1,1], speed=1, dMax=[0,0,0,0], uMode="min", dMode="max"):
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.speed = speed
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state ,spat_deriv):
        opt_a = hcl.scalar(self.uMax[0], "opt_a")
        opt_w = hcl.scalar(self.uMax[0], "opt_w")

        with hcl.if_(spat_deriv[2] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_a[0] = self.uMin[0]
        with hcl.elif_(spat_deriv[2] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_a[0] = self.uMin[0]
        #
        with hcl.if_(spat_deriv[3] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = self.uMin[1]
        with hcl.elif_(spat_deriv[3] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = self.uMin[1]
        return (opt_a[0] ,opt_w[0])

    def optDstb(self, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        v_dot = hcl.scalar(0, "v_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = state[2] * hcl.cos(state[3]) + dOpt[0]
        y_dot[0] = state[2] * hcl.sin(state[3]) + dOpt[1]
        v_dot[0] = uOpt[0] + dOpt[2]
        theta_dot[0] = uOpt[1] + dOpt[3]

        return (x_dot[0], y_dot[0], v_dot[0] ,theta_dot[0])