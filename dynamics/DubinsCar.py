import heterocl as hcl
import numpy as np
import time

class DubinsCar:
    def __init__(self, x=[0,0,0], wMax=1, speed=1, dMax=[0,0,0], uMode="min", dMode="max"):
        self.x = x
        self.wMax = wMax
        self.speed = speed
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, spat_deriv):
        opt_w = hcl.scalar(self.wMax, "opt_w")
        with hcl.if_(spat_deriv[2] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w
        with hcl.elif_(spat_deriv[2] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w
        return opt_w[0]

    def dynamics(self, theta, opt_ctrl):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = self.speed*hcl.cos(theta)
        y_dot[0] = self.speed*hcl.sin(theta)
        theta_dot[0] = opt_ctrl

        return (x_dot[0], y_dot[0], theta_dot[0])