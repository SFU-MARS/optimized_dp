import heterocl as hcl
import numpy as np

""" 6D DUBINS AIRPLANE DYNAMICS IMPLEMENTATION 
x_dot = v_x
z_dot = v_z
v_x_dot = -u_T * sin(th)
v_z_dot = u_T * cos(th) - g
th_dot = omega
omega_dot = u 
"""

class PlaneQuad6D:
    def __init__(self, x=[0,0,0,0,0,0], uMin=[-16,-5], uMax=[16,5], dMin=[0,0,0],
                 dMax=[0,0,0], uMode="min", dMode="max"):
        
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        self.g = 9.81
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
        # x_dot = v_x
        # z_dot = v_z
        # v_x_dot = -u_x * sin(th)
        # v_z_dot = u_x * cos(th) - g
        # th_dot = omega
        # omega_dot = u 

        # Graph takes in 6 possible inputs, by default
        opt_T = hcl.scalar(self.uMax[0], "opt_T")
        opt_u = hcl.scalar(self.uMax[1], "opt_u")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        in5 = hcl.scalar(0, "in5")
        in6 = hcl.scalar(0, "in6")

        if self.uMode == "min": # reaching
            with hcl.if_(spat_deriv[4] > 0):
                opt_T[0] = self.uMin[0]
            with hcl.if_(spat_deriv[5] > 0):
                opt_u[0] = self.uMin[1]
        else: # avoiding
            with hcl.if_(spat_deriv[4] < 0):
                opt_T[0] = self.uMin[0]
            with hcl.if_(spat_deriv[5] < 0):
                opt_u[0] = self.uMin[1]
        # return 4, 5, 6 even if you don't use them
        return (opt_T[0], opt_u[0], in3[0], in4[0], in5[0], in6[0])

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

        # no disturbances
        return (d1[0], d2[0], d3[0], d4[0], d5[0], d6[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        z_dot = hcl.scalar(0, "z_dot")
        v_x_dot = hcl.scalar(0, "v_x_dot")
        v_z_dot = hcl.scalar(0, "v_z_dot")
        th_dot = hcl.scalar(0, "th_dot")
        omega_dot = hcl.scalar(0, "omega_dot")

        x_dot[0] = state[2]
        z_dot[0] = state[3]
        v_x_dot[0] = -uOpt[0] * hcl.sin(state[4])
        v_z_dot[0] = uOpt[0] * hcl.cos(state[4]) - self.g
        th_dot[0] = state[5]
        omega_dot[0] = uOpt[1]

        return (x_dot[0], z_dot[0], v_x_dot[0], v_z_dot[0], th_dot[0], omega_dot[0])