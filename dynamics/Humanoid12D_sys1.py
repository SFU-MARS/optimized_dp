import heterocl as hcl
import numpy as np
import time
import math

class Humanoid12D_sys1:
    def __init__(self, x=[0,0,0,0,0,0], \
                 uMin=np.array([-50, -50, -50]), \
                 uMax=np.array([ 50,  50,  50]), \
                 dMin=np.array([0.0, 0.0, 0.0, 0.0]), \
                 dMax=np.array([0.0, 0.0, 0.0, 0.0]), \
                 dims=6, \
                 uMode="min", \
                 dMode="max" \
                 ):
        self.x = x
        self.uMode = uMode
        self.dMode = dMode

        # Object properties
        self.x    = x

        # Control bounds
        self.uMin = uMin
        self.uMax = uMax

        # Disturbance bounds
        self.dMin = dMin
        self.dMax = dMax

        self.dims = dims

        # Some constants
        self.Jxx = 5
        self.Jyy = 5
        self.Jzz = 5
        self.g = 9.81

    def dynamics(self, t, state, ctrl, dstb):
        """

        :param t: time
        :param state: tuple of grid coordinates in 6 dimensions
        :param uOpt: tuple of optimal control
        :param dOpt: tuple of optimal disturbances
        :return: tuple of time derivates in all dimensions
        """

        # System dynamics
        # x1_dot = cos(x3) / cos(x2) * x4 + sin(x3) / cos(x2) * x5
        # x2_dot = -sin(x3) * x4 + cos(x3) * x5
        # x3_dot = cos(x3) * tan(x2) * x4 + sin(x3) * tan(x2) * x5 + x6
        # x4_dot = (Jyy - Jzz) * x5 * x6 + u4 / Jxx
        # x5_dot = (Jzz - Jxx) * x4 * x6 + u5 / Jyy
        # x6_dot = (Jxx - Jyy) * x4 * x5 + u6 / Jzz

        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")
        x4_dot = hcl.scalar(0, "x4_dot")
        x5_dot = hcl.scalar(0, "x5_dot")
        x6_dot = hcl.scalar(0, "x6_dot")

        x1_dot[0] = hcl.cos(state[2]) / hcl.cos(state[1]) * state[3] + \
                    hcl.sin(state[2]) / hcl.cos(state[1]) * state[4]
        x2_dot[0] = -hcl.sin(state[2]) * state[3] + hcl.cos(state[2]) * state[4]
        x3_dot[0] = hcl.cos(state[2]) * hcl.sin(state[1]) / hcl.cos(state[1]) * state[3] + \
                    hcl.sin(state[2]) * hcl.sin(state[1]) / hcl.cos(state[1]) * state[4] + state[5]
        x4_dot[0] = (self.Jyy - self.Jzz) * state[4] * state[5] + ctrl[0] / self.Jxx
        x5_dot[0] = (self.Jzz - self.Jxx) * state[3] * state[5] + ctrl[1] / self.Jyy
        x6_dot[0] = (self.Jxx - self.Jyy) * state[3] * state[4] + ctrl[2] / self.Jzz
        
        return (x1_dot[0], x2_dot[0], x3_dot[0], x4_dot[0], x5_dot[0], x6_dot[0])

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # System dynamics
        # x1_dot = cos(x3) / cos(x2) * x4 + sin(x3) / cos(x2) * x5
        # x2_dot = -sin(x3) * x4 + cos(x3) * x5
        # x3_dot = cos(x3) * tan(x2) * x4 + sin(x3) * tan(x2) * x5 + x6
        # x4_dot = (Jyy - Jzz) * x5 * x6 + u4 / Jxx
        # x5_dot = (Jzz - Jxx) * x4 * x6 + u5 / Jyy
        # x6_dot = (Jxx - Jyy) * x4 * x5 + u6 / Jzz

        # By default, pick maximum controls
        opt_u4 = hcl.scalar(self.uMax[0], "opt_u4")
        opt_u5 = hcl.scalar(self.uMax[1], "opt_u5")
        opt_u6 = hcl.scalar(self.uMax[2], "opt_u6")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[3] > 0):
                opt_u4[0] = self.uMin[0]
            with hcl.if_(spat_deriv[4] > 0):
                opt_u5[0] = self.uMin[1]
            with hcl.if_(spat_deriv[5] > 0):
                opt_u6[0] = self.uMin[2]
        else: # self.uMode == "max"
            with hcl.if_(spat_deriv[3] < 0):
                opt_u4[0] = self.uMin[0]
            with hcl.if_(spat_deriv[4] < 0):
                opt_u5[0] = self.uMin[1]
            with hcl.if_(spat_deriv[5] < 0):
                opt_u6[0] = self.uMin[2]
                
        return (opt_u4[0], opt_u5[0], opt_u6[0])

    def optDstb(self, spat_deriv):
        """
        :param spat_deriv: spatial derivative in all dimensions
        :return: tuple of optimal disturbance
        """
        dOpt1 = hcl.scalar(0, "dOpt1")
        dOpt2 = hcl.scalar(0, "dOpt2")
        dOpt3 = hcl.scalar(0, "dOpt3")
        dOpt4 = hcl.scalar(0, "dOpt4")
        dOpt5 = hcl.scalar(0, "dOpt5")
        dOpt6 = hcl.scalar(0, "dOpt6")
        return (dOpt1[0], dOpt2[0], dOpt3[0], dOpt4[0], dOpt5[0], dOpt6[0])

