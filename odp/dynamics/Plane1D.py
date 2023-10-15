import heterocl as hcl
import numpy as np

class Plane1D:
    # Dynamics: x_dot = vx, y_dot = vy
    #          vx_dot \in [vxMin, vxMax], vy_dot \in [vyMin, vyMax]
    def __init__(self, x=[0], vMin=-1, vMax=1, dMin=0, dMax=0, uMode="min", dMode="max"):
        self.x = x
        self.vMin = vMin
        self.vMax = vMax
        self.dMin = dMin
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_v = hcl.scalar(self.vMax, "opt_v")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")

        with hcl.if_(spat_deriv[0] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_v[0] = self.vMin
        with hcl.elif_(spat_deriv[0] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_v[0] = self.vMin

        return (opt_v[0], )

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")

        # Just create and pass back, even though they're not used

        return (d1[0], )

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")

        x_dot[0] = uOpt[0] + dOpt[0]

        return (x_dot[0], )
