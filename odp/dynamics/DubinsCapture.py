import heterocl as hcl

class DubinsCapture:
    def __init__(self, x=[0,0,0], wMax=1.0, speed=1.0, dMax=1.0, uMode="max", dMode="min"):
        self.x = x
        self.wMax = wMax
        self.speed = speed
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode
        self.ctrl_dims = 1
        self.dstb_dims = 1
        self.state_dims = 3

    def opt_ctrl(self, u, dv, t, x):


        #opt_w = hcl.scalar(self.wMax, "opt_w")
        u[0] = self.wMax

        a_term = hcl.scalar(0, "a_term")
        a_term[0] = dv[0] * x[1] - dv[1] * x[0] - dv[2]

        with hcl.if_(a_term >= 0):
            with hcl.if_(self.uMode == "min"):
                u[0] = -self.wMax
        with hcl.elif_(a_term < 0):
            with hcl.if_(self.uMode == "max"):
                u[0] = -self.wMax

    def opt_dstb(self, d, dv, t, x):
        d[0] = self.dMax
        with hcl.if_(dv[2] >= 0):
            with hcl.if_(self.dMode == "min"):
                d[0] = -self.dMax
        with hcl.elif_(dv[2] < 0):
            with hcl.if_(self.dMode == "max"):
                d[0] = -self.dMax

    def dynamics(self, dx, t, x, u, d):
        dx[0] = -self.speed + self.speed*hcl.cos(x[2]) + u[0]*x[1]
        dx[1] = self.speed*hcl.sin(x[2]) - u[0]*x[0]
        dx[2] = u[0] - d[0]

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