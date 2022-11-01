import heterocl as hcl
import math

""" 4D 1v1 AttackerDefender DYNAMICS IMPLEMENTATION 
 xA1_dot = vA * u1
 xA2_dot = vA * u2
 xD1_dot = vD * d1
 xD2_dot = vD * d2
 """


class AttackerDefender4D:
    def __init__(self, x=[0, 0, 0, 0], uMin=[-1, -1], uMax=[1, 1], dMin=[-1, -1],
                 dMax=[1, 1], uMode="min", dMode="max"):
        """Creates a Dublin Car with the following states:
           X position, Y position, speed, heading
           The controls are the acceleration and turn rate (angular speed)
           The disturbances are the noise in the velocity components.
        Args:
            x (list, optional): Initial state . Defaults to [0,0,0,0].
            uMin (list, optional): Lowerbound of user control. Defaults to [-1,-1].
            uMax (list, optional): Upperbound of user control.
                                   Defaults to [1,1].
            dMin (list, optional): Lowerbound of disturbance to user control, . Defaults to [-0.25,-0.25].
            dMax (list, optional): Upperbound of disturbance to user control. Defaults to [0.25,0.25].
            uMode (str, optional): Accepts either "min" or "max".
                                   * "min" : have optimal control reach goal
                                   * "max" : have optimal control avoid goal
                                   Defaults to "min".
            dMode (str, optional): Accepts whether "min" or "max" and should be opposite of uMode.
                                   Defaults to "max".
        """
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        assert (uMode in ["min", "max"])
        self.uMode = uMode
        if uMode == "min":
            assert (dMode == "max")
        else:
            assert (dMode == "min")
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # System dynamics
        # xA1_dot = vA * u1
        # xA2_dot = vA * u2
        # xD1_dot = vD * d1
        # xD2_dot = vD * d2

        # Graph takes in 4 possible inputs, by default, for now
        # In 1v1AttackerDefender, a(t) = [a1, a2]^T
        opt_a1 = hcl.scalar(self.uMax[0], "opt_a1")
        opt_a2 = hcl.scalar(self.uMax[1], "opt_a2")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[0] > 0):
                opt_a1[0] = self.uMin[0]  # now is Bang-bang control and I should revise it into equation (11)
            with hcl.if_(spat_deriv[1] > 0):
                opt_a2[0] = self.uMin[1]
        else:
            with hcl.if_(spat_deriv[0] < 0):
                opt_a1[0] = self.uMin[0]
            with hcl.if_(spat_deriv[1] < 0):
                opt_a2[0] = self.uMin[1]
        # return 3, 4 even if you don't use them
        return opt_a1[0], opt_a2[0], in3[0], in4[0]

    def opt_dstb(self, t, state, spat_deriv):
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

        # with hcl.if_(self.dMode == "max"):
        # I should clarify the expression the d(t) using equation (12)
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
        # maximum velocity
        vA = hcl.scalar(1.0, "vA")
        vD = hcl.scalar(1.0, "vD")

        xA1_dot = hcl.scalar(0, "xA1_dot")
        xA2_dot = hcl.scalar(0, "xA2_dot")
        xD1_dot = hcl.scalar(0, "xD1_dot")
        xD2_dot = hcl.scalar(0, "xD2_dot")

        xA1_dot[0] = vA * uOpt[0]
        xA2_dot[0] = vA * uOpt[1]
        xD1_dot[0] = vD * dOpt[0]
        xD2_dot[0] = vD * dOpt[1]

        return xA1_dot[0], xA2_dot[0], xD1_dot[0], xD2_dot[0]

        # The below function can have whatever form or parameters users want
        # These functions are not used in HeteroCL program, hence is pure Python code and
        # can be used after the value function has been obtained.

        def optCtrl_inPython(self, spat_deriv):
            """
            :param t: time t
            :param state: tuple of coordinates
            :param spat_deriv: tuple of spatial derivative in all dimensions
            :return:
            """
        # System dynamics
        # xA1_dot = vA * u1
        # xA2_dot = vA * u2
        # xD1_dot = vD * d1
        # xD2_dot = vD * d2
        opt_a = self.uMax[0]
        opt_w = self.uMax[1]

        # The initialized control only change sign in the following cases
        if self.uMode == "min":
            if spat_deriv[2] > 0:
                opt_a = self.uMin[0]
            if spat_deriv[3] > 0:
                opt_w = self.uMin[1]
        else:
            if spat_deriv[2] < 0:
                opt_a = self.uMin[0]
            if spat_deriv[3] < 0:
                opt_w = self.uMin[1]

        return opt_a, opt_w
