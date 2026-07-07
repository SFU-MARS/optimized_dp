import heterocl as hcl
import numpy as np

""" 6D Two Dubins Airplane stacked dynamics. 
 x1_dot = v1 * cos(theta1)
 y1_dot = v1 * sin(theta1)
 theta1_dot = omega_1 

 x2_dot = v2 * cos(theta2)
 y2_dot = v2 * sin(theta2)
 theta2_dot = omega_2 
 """

class TwoDubins3D:
    def __init__(self, x=[0,0,0,0,0,0], uMin=[-1,-1], uMax=[1,1], dMin=[0., 0.],
                 dMax=[0., 0.], uMode="min", dMode="max", v=1.0):
        
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin

        self.v1 = v
        self.v2 = v
        assert(uMode in ["min", "max"])
        self.uMode = uMode
        if uMode == "min":
            assert(dMode == "max")
        else:
            assert(dMode == "min")
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # System dynamics
        # x1_dot = v1 * cos(theta1)
        # y1_dot = v1 * sin(theta1)
        # theta1_dot = omega_1 

        # x2_dot = v2 * cos(theta2)
        # y2_dot = v2 * sin(theta2)
        # theta2_dot = omega_2 

        # Graph takes in 6 possible inputs, by default, for now
        omega_1 = hcl.scalar(self.uMax[0], "omega_1")
        omega_2 = hcl.scalar(self.uMax[1], "omega_2")
        

        if self.uMode == "min":
            with hcl.if_(spat_deriv[2] > 0):
                omega_1[0] = self.uMin[0]
            with hcl.if_(spat_deriv[5] > 0):
                omega_2[0] = self.uMin[1]
        else:
            with hcl.if_(spat_deriv[2] < 0):
                omega_1[0] = self.uMin[0]
            with hcl.if_(spat_deriv[5] < 0):
                omega_2[0] = self.uMin[1]
        
        # Just create and pass back dummy values
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        in5 = hcl.scalar(0, "in5")
        in6 = hcl.scalar(0, "in6")

        # return 4, 5, 6 even if you don't use them
        return (omega_1[0], omega_2[0], in3[0], in4[0], in5[0], in6[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 6 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        # Just create and pass back, even though they're not used
        d4 = hcl.scalar(0, "d4")
        d5 = hcl.scalar(0, "d5")
        d6 = hcl.scalar(0, "d6")

        return (d1[0], d2[0], d3[0], d4[0], d5[0], d6[0])

    def dynamics(self, t, state, uOpt, dOpt):
        """
        2 3D Dubins Car stacked dynamics.
        x1_dot = v1 * cos(theta1)
        y1_dot = v1 * sin(theta1)
        theta1_dot = omega_1 

        x2_dot = v2 * cos(theta2)
        y2_dot = v2 * sin(theta2)
        theta2_dot = omega_2 
        """        

        x1_dot = hcl.scalar(0, "x1_dot")
        y1_dot = hcl.scalar(0, "y1_dot")
        theta1_dot = hcl.scalar(0, "theta1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        y2_dot = hcl.scalar(0, "y2_dot")
        theta2_dot = hcl.scalar(0, "theta2_dot")

        x1_dot[0] = self.v1 * hcl.cos(state[2])
        y1_dot[0] = self.v1 * hcl.sin(state[2])
        theta1_dot[0] = uOpt[0]
        x2_dot[0] = self.v2 * hcl.cos(state[5])
        y2_dot[0] = self.v2 * hcl.sin(state[5])
        theta2_dot[0] = uOpt[1]

        return (x1_dot[0], y1_dot[0], theta1_dot[0], x2_dot[0], y2_dot[0], theta2_dot[0])
