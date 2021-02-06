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
# System dynamics (state constraints and discretization)
# p.theta_dim = int(30)
# p.v_dim = int(9)
# p.v_dv = float(.1)
# p.wMax = float(1.1)
# p.aMax = float(.4)
# p.v_high = float(.7)
# p.v_low = float(-.1)
#
# # System dynamics (disturbances constraints)
# # Additive disturbances
# # Disturbances used for modeling prediction errors
# p.dMax_xy = float(0.05)
# p.dMax_theta = float(0.15)
# p.dMax_avoid_xy = float(0.05)
# p.dMax_avoid_theta = float(0.15)

class DubinsCar4Davoid:
    def __init__(self, x=[0,0,0,0], uMin = [-0.4,-1.1], uMax = [0.4,1.1], dMin = [-0.05,-0.15], \
                 dMax=[0.05,0.15], uMode="max", dMode="min",v_max=[1], t_step=[0.01]):
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        self.uMode = uMode
        self.dMode = dMode
        self.t_step=t_step
        self.v_max=v_max

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
        t_step=hcl.scalar(self.t_step[0], "t_step")
        v_max=hcl.scalar(self.v_max[0], "v_max")
        # Just create and pass back, even though they're not used
        in3   = hcl.scalar(0, "in3")
        in4   = hcl.scalar(0, "in4")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[2] > 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(spat_deriv[3] > 0):
                opt_w[0] = self.uMin[1]
        else:
            with hcl.if_(spat_deriv[2] > 0):
                # opt_a[0] = self.uMax[0]
                # opt_a[0] = np.min(self.uMax[0],(v_max-state[2])/t_step)
                # accel_lim = hcl.scalar(0, "accel_lim")
                # accel_lim[0] = (v_max - state[2]) / t_step
                opt_a[0] = (v_max - state[2]) / t_step
                with hcl.if_(opt_a[0] > self.uMax[0]):
                    opt_a[0] = self.uMax[0]
            with hcl.if_(spat_deriv[2] < 0):
                # opt_a[0] = self.uMin[0]
                # opt_a[0] = np.max(self.uMin[0],state[2]/t_step)
                # accel_lim1 = hcl.scalar(0, "accel_lim")
                # accel_lim1[0] = state[2] / t_step
                opt_a[0] = state[2] / t_step
                with hcl.if_(opt_a[0] < self.uMin[0]):
                    opt_a[0] = self.uMin[0]

            with hcl.if_(spat_deriv[3] < 0):
                opt_w[0] = self.uMin[1]
	    #with hcl.if_(spat_deriv[3] > 0):
	       # opt_w[0] = self.umax[1]
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

        cos_theta = hcl.scalar(0, "cos_theta")
        sin_theta = hcl.scalar(0,"sin_theta")
        spat1_sq = hcl.scalar(0, "spat1_sq")
        spat2_sq = hcl.scalar(0,"spat2_sq")
        sum_v = hcl.scalar(0,"sum_v")
        norm = hcl.scalar(0,"norm")
        spat1_sq[0] = spat_deriv[0] * spat_deriv[0]
        spat2_sq[0] = spat_deriv[1] * spat_deriv[1]
        sum_v[0]    = spat1_sq[0] + spat2_sq[0]
        norm[0]	    = hcl.sqrt(sum_v[0])
        cos_theta[0] = spat_deriv[0] / (norm[0] + 0.0001)
        sin_theta[0] = spat_deriv[1] / (norm[0] + 0.0001)
        #with hcl.if_(self.dMode == "max"):
        if self.dMode == "max":
            with hcl.if_(spat_deriv[2] > 0):
                d1[0] = self.dMax[0]
            with hcl.elif_(spat_deriv[2] < 0):
                d1[0] = self.dMin[0]
            with hcl.if_(spat_deriv[3] > 0):
                d2[0] = self.dMax[1]
            with hcl.elif_(spat_deriv[3] < 0):
                d2[0] = self.dMin[1]
            # with hcl.elif_(spat_deriv[0] > 0):
            #     d3[0] = - cos_theta
            # with hcl.elif_(spat_deriv[0] < 0):
            #     d3[0] = cos_theta
            # with hcl.elif_(spat_deriv[1] > 0):
            #     d4[0] = - sin_theta
            # with hcl.elif_(spat_deriv[1] < 0):
            #     d4[0] = sin_theta
            d3[0] = -cos_theta[0]
            d4[0] = -sin_theta[0]

        else:
            # with hcl.if_(spat_deriv[2] > 0):
            #     d1[0] = self.dMin[0]
            # with hcl.elif_(spat_deriv[2] < 0):
            #     d1[0] = self.dMax[0]
            with hcl.if_(spat_deriv[3] > 0):
                d4[0] = -self.dMax[1]
            with hcl.elif_(spat_deriv[3] < 0):
                d4[0] = self.dMax[1]
            # with hcl.elif_(spat_deriv[0] > 0):
            #     d3[0] = cos_theta
            # with hcl.elif_(spat_deriv[0] < 0):
            #     d3[0] = - cos_theta
            # with hcl.elif_(spat_deriv[1] > 0):
            #     d4[0] = sin_theta
            # with hcl.elif_(spat_deriv[1] < 0):
            #     d4[0] = - sin_theta
            d1[0] = cos_theta[0]*-self.dMax[0]
            d2[0] = sin_theta[0]*-self.dMax[0]
            d3[0]=0
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
