import heterocl as hcl
import numpy as np

from .base import DynamicsBase

__all__ = ['Bicycle5D']


class Bicycle5D(DynamicsBase):

    state_dims = 5
    ctrl_dims = 2
    dstb_dims = state_dims

    wheelbase = 0.32

    X, Y, YAW, DELTA, VEL = range(state_dims)
    STEERING, ACCELERATION = range(ctrl_dims)

    def __init__(self, ctrl_range, dstb_range, mode='reach', *, wheelbase):
        super().__init__(ctrl_range, dstb_range, mode)

        self.wheelbase = wheelbase

    def dynamics(self, dx, t, x, u, d):

        # x_dot = v * cos(yaw) + d_1
        dx[self.X] = (x[self.VEL] * hcl.cos(x[self.YAW]) 
                      + d[self.X])

        # y_dot = v * sin(yaw) + d_2
        dx[self.Y] = (x[self.VEL] * hcl.sin(x[self.YAW])
                      + d[self.Y])

        # yaw_dot = (v * tan(delta))/L + d3
        hcl_tan = lambda a: hcl.sin(a) / hcl.cos(a)
        dx[self.YAW] = (x[self.VEL] * hcl_tan(x[self.DELTA])/self.wheelbase
                        + d[self.YAW])

        # delta_dot = u1 + d4
        dx[self.DELTA] = u[self.STEERING] # + d[self.DELTA]

        # v_dot = u2 + d5
        dx[self.VEL] = u[self.ACCELERATION] # + d[self.VEL]

    def opt_ctrl(self, u, dv, t, x):

        uMin, uMax = self.ctrl_range
        
        if self.mode['uMode'] == "max":
            # Steering
            with hcl.if_(0 <= x[self.VEL]):
                with hcl.if_(0 <= dv[self.YAW]):
                    u[self.STEERING] = uMax[self.STEERING]
                with hcl.else_():
                    u[self.STEERING] = uMin[self.STEERING]
            with hcl.else_():
                with hcl.if_(0 <= dv[self.YAW]):
                    u[self.STEERING] = uMin[self.STEERING]
                with hcl.else_():
                    u[self.STEERING] = uMax[self.STEERING]
            # Velocity
            with hcl.if_(0 <= dv[self.VEL]):
                u[self.ACCELERATION] = uMax[self.ACCELERATION]
            with hcl.else_():
                u[self.ACCELERATION] = uMin[self.ACCELERATION]            
        else:
            # Steering
            with hcl.if_(0 <= x[self.VEL]):
                with hcl.if_(0 <= dv[self.YAW]):
                    u[self.STEERING] = uMin[self.STEERING]
                with hcl.else_():
                    u[self.STEERING] = uMax[self.STEERING]
            with hcl.else_():
                with hcl.if_(0 <= dv[self.YAW]):
                    u[self.STEERING] = uMax[self.STEERING]
                with hcl.else_():
                    u[self.STEERING] = uMin[self.STEERING]
            # Velocity
            with hcl.if_(0 <= dv[self.VEL]):
                u[self.ACCELERATION] = uMin[self.ACCELERATION]
            with hcl.else_():
                u[self.ACCELERATION] = uMax[self.ACCELERATION]

    def opt_dstb(self, d, dv, t, x):

        dMin, dMax = self.dstb_range

        for i in range(self.dstb_dims):
            if self.mode['dMode'] == "max":
                with hcl.if_(0 <= dv[i]):
                    d[i] = dMax[i]
                with hcl.else_():
                    d[i] = dMin[i]
            else:
                with hcl.if_(0 <= dv[i]):
                    d[i] = dMin[i]
                with hcl.else_():
                    d[i] = dMax[i]
