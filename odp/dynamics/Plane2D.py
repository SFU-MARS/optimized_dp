import heterocl as hcl
import numpy as np

class Plane2D:
    # Dynamics: x_dot = vx, y_dot = vy
    #          vx_dot \in [vxMin, vxMax], vy_dot \in [vyMin, vyMax]
    def __init__(self, x=[0,0], vxMin=-1, vxMax=1, vyMin=-1, vyMax=1, uMode="min"):
        self.x = x
        self.vxMin = vxMin
        self.vxMax = vxMax
        self.vyMin = vyMin
        self.vyMax = vyMax
        self.uMode = uMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_vx = hcl.scalar(self.vxMax, "opt_vx")
        opt_vy = hcl.scalar(self.vyMax, "opt_vy")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")

        with hcl.if_(spat_deriv[0] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_vx[0] = self.vxMin
        with hcl.elif_(spat_deriv[0] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_vx[0] = self.vxMin

        with hcl.if_(spat_deriv[1] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_vy[0] = self.vyMin 
        with hcl.elif_(spat_deriv[1] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_vy[0] = self.vyMin

        return (opt_vx[0], opt_vy[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        return (d1[0], d2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")

        x_dot[0] = uOpt[0]
        y_dot[0] = uOpt[1]

        return (x_dot[0], y_dot[0])
