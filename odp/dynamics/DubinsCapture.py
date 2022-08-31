import heterocl as hcl

class DubinsCapture:
    def __init__(self, x=[0,0,0], wMax=1.0, speed=1.0, dMax=1.0, uMode="max", dMode="min"):
        self.x = x
        self.wMax = wMax
        self.speed = speed
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
                :param  spat_deriv: tuple of spatial derivative in all dimensions
                        state: x1, x2, x3
                        t: time
                :return: a tuple of optimal disturbances
        """

        opt_w = hcl.scalar(self.wMax, "opt_w")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        #a_term = spat_deriv[0] * self.x[1] - spat_deriv[1]*self.x[0] - spat_deriv[2]

        # Declare a variable
        a_term = hcl.scalar(0, "a_term")
        # use the scalar by indexing 0 everytime
        a_term[0] = spat_deriv[0] * state[1] - spat_deriv[1] * state[0] - spat_deriv[2]

        with hcl.if_(a_term >= 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w[0]
        with hcl.elif_(a_term < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w[0]
        return (opt_w[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
            :param spat_deriv: tuple of spatial derivative in all dimensions
                    state: x1, x2, x3
                    t: time
            :return: a tuple of optimal disturbances
        """

        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(self.dMax, "d1")
        d2 = hcl.scalar(0, "d2")
        # Just create and pass back, even though they're not used
        d3 = hcl.scalar(0, "d3")

        # Declare a variable
        b_term = hcl.scalar(0, "b_term")
        # use the scalar by indexing 0 everytime
        b_term[0] = spat_deriv[2]

        with hcl.if_(b_term[0] >= 0):
            with hcl.if_(self.dMode == "min"):
                d1[0] = -d1[0]
        with hcl.elif_(b_term[0] < 0):
            with hcl.if_(self.dMode == "max"):
                d1[0] = -d1[0]
        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = -self.speed + self.speed*hcl.cos(state[2]) + uOpt[0]*state[1]
        y_dot[0] = self.speed*hcl.sin(state[2]) - uOpt[0]*state[0]
        theta_dot[0] = dOpt[0] - uOpt[0]

        return (x_dot[0], y_dot[0], theta_dot[0])

    # The below function can have whatever form or parameters users want
    # These functions are not used in HeteroCL program, hence is pure Python code and
    # can be used after the value function has been obtained.
    def optCtrl_inPython(self, state, spat_deriv):
        a_term = spat_deriv[0] * state[1] - spat_deriv[1] * state[0] - spat_deriv[2]

        opt_w = self.wMax
        if a_term >= 0:
            if self.uMode == "min":
                opt_w = -self.wMax
        else:
            if self.uMode == "max":
                opt_w = -self.wMax
        return opt_w