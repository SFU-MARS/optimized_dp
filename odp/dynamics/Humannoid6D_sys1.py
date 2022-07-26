import heterocl as hcl
import numpy as np

class Humanoid_6D:
    def __init__(self, x=[0, 0, 0, 0, 0, 0], uMin=np.array([-0.5*0.1, -5.0, -1.0]), uMax=np.array([0.5*0.1, 5.0, 1.0]),
                 dMin=np.array([0.0, 0.0, 0.0, 0.0]), dMax=np.array([0.0, 0.0, 0.0, 0.0]),
                 dims=6, uMode="min", dMode="max"):
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
        self.r_f = 0.1
        self.a_z_max = 5
        self.a_x_max = 1
        self.J = 0.125
        self.g = -9.81
        self.L = 1.2

    def dynamics(self, t,state, uOpt, dOpt):
        """

        :param t: time
        :param state: tuple of grid coordinates in 6 dimensions
        :param uOpt: tuple of optimal control
        :param dOpt: tuple of optimal disturbances
        :return: tuple of time derivates in all dimensions
        """

        # Set of differential equations describing the system
        # x1_dot = x2
        # x2_dot = ( g + u2) * (x1 + u1) /x3 + u3
        # x3_dot = x4
        # x4_dot = u2
        # x5_dot = x6
        # x6_dot = ( x3 / J ) * u3

        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")
        x4_dot = hcl.scalar(0, "x4_dot")
        x5_dot = hcl.scalar(0, "x5_dot")
        x6_dot = hcl.scalar(0, "x6_dot")

        x1_dot[0] = state[1]
        x2_dot[0] = (self.g + uOpt[1])*(state[0] + uOpt[0])/state[2] + uOpt[2]
        x3_dot[0] = state[3]
        x4_dot[0] = uOpt[1]
        x5_dot[0] = state[5]
        x6_dot[0] = state[2] * uOpt[2]/ self.J
        return (x1_dot[0], x2_dot[0], x3_dot[0], x4_dot[0], x5_dot[0], x6_dot[0])

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

        SumUU = hcl.scalar(0, "SumUU")
        SumUL = hcl.scalar(0, "SumUL")
        SumLU = hcl.scalar(0, "SumLU")
        SumLL = hcl.scalar(0, "SumLL")
        parSum= hcl.scalar(0, "parSum")
        
        with hcl.if_(self.uMode == "min"):
            # everything containing (u1, u2) in Hamiltonian, for all combinations of u1 and u2
            SumUU[0] = spat_deriv[1]*(self.g + self.uMax[1]) * (state[0] + self.uMax[0])/state[2] + \
                       spat_deriv[3] * self.uMax[1]
            SumUL[0] = spat_deriv[1]*(self.g + self.uMax[1]) * (state[0] + self.uMin[0])/state[2] + \
                       spat_deriv[3] * self.uMax[1]
            SumLU[0] = spat_deriv[1]*(self.g + self.uMin[1]) * (state[0] + self.uMax[0])/state[2] + \
                       spat_deriv[3] * self.uMin[1]
            SumLL[0] = spat_deriv[1]*(self.g + self.uMin[1]) * (state[0] + self.uMin[0])/state[2] + \
                       spat_deriv[3] * self.uMin[1]

            # try every combination of u1 and u2 and take minimum
            with hcl.if_(SumUU[0] > SumUL[0]):
                uOpt1[0] = self.uMin[0]
                uOpt2[0] = self.uMax[1]
                SumUU[0] = SumUL[0]
                
            with hcl.elif_(SumUU[0] < SumUL[0]):
                uOpt1[0] = self.uMax[0]
                uOpt2[0] = self.uMax[1]
                
            with hcl.if_(SumUU[0] > SumLU[0]):
                uOpt1[0] = self.uMax[0]
                uOpt2[0] = self.uMin[1]
                SumUU[0] = SumLU[0]

            with hcl.if_(SumUU[0] > SumLL[0]):
                uOpt1[0] = self.uMin[0]
                uOpt2[0] = self.uMin[1]

            # Find u3
            # everything multiplied by u3 in Hamiltonian
            parSum[0] = spat_deriv[1] + spat_deriv[5]*state[2]/self.J 
            
            with hcl.if_(parSum[0] > 0):
                uOpt3[0] = self.uMin[2]
                
            with hcl.elif_(parSum[0] < 0):
                uOpt3[0] = self.uMax[2]
                
        return (uOpt1[0], uOpt2[0], uOpt3[0])

    def opt_dstb(self, spat_deriv):
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

