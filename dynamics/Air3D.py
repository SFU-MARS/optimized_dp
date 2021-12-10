import heterocl as hcl
import numpy as np
import time

class Air3D:
    def __init__(self, x=[0,0,0], omega_max=1.1, velocity=0.75, angle_alpha = 1.0, dMax=[0,0,0.0], uMode="min", dMode="max"):
        self.x = x
        # assume that both systems use [-omega_max, omega_max] their control 
        self.omega_max = omega_max
        self.velocity = velocity
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.omega_max, "opt_w")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        det = hcl.scalar(0, 'det')
        det[0] = spat_deriv[0] * state[1] - spat_deriv[1] * state[0] - spat_deriv[2]

        with hcl.if_(self.uMode == 'max'):
            with hcl.if_(det < 0):
                    opt_w[0] = -opt_w

        with hcl.if_(self.uMode == "min"):
            with hcl.if_(det > 0):
                    opt_w[0] = -opt_w

        return opt_w[0], in3[0], in4[0]

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 3 possible inputs, by default, for now
        opt_d = hcl.scalar(self.omega_max, "opt_d")
        d2 = hcl.scalar(0, "d2")
        # Just create and pass back, even though they're not used
        d3 = hcl.scalar(0, "d3")

        with hcl.if_(self.uMode == "max"):
            with hcl.if_(spat_deriv[2] < 0):
                    opt_d[0] = -opt_d

        with hcl.if_(self.uMode == "min"):
            with hcl.if_(spat_deriv[2] >= 0):
                    opt_d[0] = -opt_d


        return opt_d[0], d2[0], d3[0]

    def dynamics(self, t, state, uOpt, dOpt):
        """
        \dot x = -v_a + v_b \cos \psi + a y
        \dot y = v_b \sin \psi + a y
        \dot \psi = b- a
        """
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = -self.velocity + self.velocity * hcl.cos(state[2]) + uOpt[0] * state[1]
        y_dot[0] = self.velocity * hcl.sin(state[2]) - uOpt[0] * state[0]
        theta_dot[0] = dOpt[0] - uOpt[0]

        return x_dot[0], y_dot[0], theta_dot[0]
