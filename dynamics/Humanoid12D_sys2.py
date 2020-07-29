import heterocl as hcl
import numpy as np
import time
import math

class Humanoid12D_sys2:
    def __init__(self, x=[0,0,0,0,0,0], \
                 uMin=np.array([-0.05, -0.05, 5]), \
                 uMax=np.array([0.05, 0.05, 15]), \
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

        # Constants
        self.g = 9.81

    def dynamics(self, t,state, ctrl, dstb):
        """

        :param t: time
        :param state: tuple of grid coordinates in 6 dimensions
        :param uOpt: tuple of optimal control
        :param dOpt: tuple of optimal disturbances
        :return: tuple of time derivates in all dimensions
        """

        # Dynamics
        # x7_dot = x10
        # x8_dot = x11
        # x9_dot = x12
        # x10_dot = u3 * (x7 - u1)
        # x11_dot = u3 * (x8 - u2)
        # x12_dot = u3 * x9 - g

        x7_dot = hcl.scalar(0, "x7_dot")
        x8_dot = hcl.scalar(0, "x8_dot")
        x9_dot = hcl.scalar(0, "x9_dot")
        x10_dot = hcl.scalar(0, "x10_dot")
        x11_dot = hcl.scalar(0, "x11_dot")
        x12_dot = hcl.scalar(0, "x12_dot")

        x7_dot[0] = state[3]
        x8_dot[0] = state[4]
        x9_dot[0] = state[5]
        x10_dot[0] = ctrl[2] * (state[0] - ctrl[0])
        x11_dot[0] = ctrl[2] * (state[1] - ctrl[1])
        x12_dot[0] = ctrl[2] * state[2] - self.g
        
        return (x7_dot[0], x8_dot[0], x9_dot[0], x10_dot[0], x11_dot[0], x12_dot[0])

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates in 6 dimensions
        :param spat_deriv: spatial derivative in all dimensions
        :return: tuple of optimal control
        """

        # Optimal control 1, 2, 3
        uOpt1 = hcl.scalar(0, "uOpt1")
        uOpt2 = hcl.scalar(0, "uOpt2")
        uOpt3 = hcl.scalar(0, "uOpt3")
        
        uOpt3UU = hcl.scalar(0, "uOpt3UU")
        uOpt3UL = hcl.scalar(0, "uOpt3UL")
        uOpt3LU = hcl.scalar(0, "uOpt3LU")
        uOpt3LL = hcl.scalar(0, "uOpt3LL")
        
        SumUU = hcl.scalar(0, "SumUU")
        SumUL = hcl.scalar(0, "SumUL")
        SumLU = hcl.scalar(0, "SumLU")
        SumLL = hcl.scalar(0, "SumLL")
        
        with hcl.if_(self.uMode == "min"):
            # Ham value over all combinations of (u1, u2), ignoring u3
            # In each case u3 is determined from (u1, u2)
            SumUU[0] = spat_deriv[3] * (state[0] - self.uMax[0]) + \
                       spat_deriv[4] * (state[1] - self.uMax[1]) + \
                       spat_deriv[5] * state[2]
            with hcl.if_(SumUU[0] > 0): # Pick u3
                uOpt3UU[0] = self.uMin[2]
            with hcl.else_():
                uOpt3UU[0] = self.uMax[2]
            SumUU[0] = SumUU[0] * uOpt3UU[0]
                
            SumUL[0] = spat_deriv[3] * (state[0] - self.uMax[0]) + \
                       spat_deriv[4] * (state[1] - self.uMin[1]) + \
                       spat_deriv[5] * state[2]
            with hcl.if_(SumUL[0] > 0): # Pick u3
                uOpt3UL[0] = self.uMin[2]
            with hcl.else_():
                uOpt3UL[0] = self.uMax[2]
            SumUL[0] = SumUL[0] * uOpt3UL[0]        
                
            SumLU[0] = spat_deriv[3] * (state[0] - self.uMin[0]) + \
                       spat_deriv[4] * (state[1] - self.uMax[1]) + \
                       spat_deriv[5] * state[2]
            with hcl.if_(SumLU[0] > 0): # Pick u3
                uOpt3LU[0] = self.uMin[2]
            with hcl.else_():
                uOpt3LU[0] = self.uMax[2]
            SumLU[0] = SumLU[0] * uOpt3LU[0]  
                
            SumLL[0] = spat_deriv[3] * (state[0] - self.uMin[0]) + \
                       spat_deriv[4] * (state[1] - self.uMin[1]) + \
                       spat_deriv[5] * state[2]
            with hcl.if_(SumLL[0] > 0): # Pick u3
                uOpt3LL[0] = self.uMin[2]
            with hcl.else_():
                uOpt3LL[0] = self.uMax[2]

            SumLL[0] = SumLL[0] * uOpt3LL[0]

            # Go through Hamiltonian value for each case to pick the best (u1, u2, u3)
            uOpt1[0] = self.uMax[0]
            uOpt2[0] = self.uMax[1]
            uOpt3[0] = uOpt3UU[0]
                
            with hcl.if_(SumUL[0] < SumUU[0]):
                uOpt1[0] = self.uMax[0]
                uOpt2[0] = self.uMin[1]
                uOpt3[0] = uOpt3UL
                SumUU[0] = SumUL[0]
                
            with hcl.if_(SumLU[0] < SumUU[0]):
                uOpt1[0] = self.uMin[0]
                uOpt2[0] = self.uMax[1]
                uOpt3[0] = uOpt3LU
                SumUU[0] = SumLU[0]

            with hcl.if_(SumLL[0] < SumUU[0]):
                uOpt1[0] = self.uMin[0]
                uOpt2[0] = self.uMin[1]
                uOpt3[0] = uOpt3LL
                
        return (uOpt1[0], uOpt2[0], uOpt3[0])

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

