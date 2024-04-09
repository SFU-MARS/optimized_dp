import heterocl as hcl
import numpy as np

from .base import DynamicsBase

__all__ = ['Bicycle4D']


class Bicycle4D(DynamicsBase):

    state_dims = 4 
    ctrl_dims = 2 
    dstb_dims = state_dims

    wheelbase = 0.32

    X, Y, YAW, VEL = range(state_dims)
    STEERING, VELOCITY = range(ctrl_dims)

    def __init__(self, ctrl_range, dstb_range, mode='reach', *, wheelbase):

        self.wheelbase = wheelbase

        self.ctrl_range = np.asarray(ctrl_range)
        assert self.ctrl_range.shape[1] == self.ctrl_dims

        self.dstb_range = np.asarray(dstb_range)
        assert self.dstb_range.shape[1] == self.dstb_dims

        modes = {'reach': {"uMode": "min", "dMode": "max"},
                 'avoid': {"uMode": "max", "dMode": "min"}}
        self.mode = modes[mode]

    def dynamics(self, dx, t, x, u, d):

        # x_dot = v * cos(theta) + d_1
        dx[self.X] = (x[self.VEL] * hcl.cos(x[self.YAW]) 
                      + d[self.X])

        # y_dot = v * sin(theta) + d_2
        dx[self.Y] = (x[self.VEL] * hcl.sin(x[self.YAW])
                      + d[self.Y])

        # theta_dot = (v * tan(u1))/L + d3
        hcl_tan = lambda a: hcl.sin(a) / hcl.cos(a)
        dx[self.YAW] = (x[self.VEL] * hcl_tan(u[self.STEERING])/self.wheelbase
                        + d[self.YAW])

        # v_dot = u2 + d4
        dx[self.VEL] = u[self.VELOCITY] + d[self.VEL]

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
                u[self.VELOCITY] = uMax[self.VELOCITY]
            with hcl.else_():
                u[self.VELOCITY] = uMin[self.VELOCITY]            
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
                u[self.VELOCITY] = uMin[self.VELOCITY]
            with hcl.else_():
                u[self.VELOCITY] = uMax[self.VELOCITY]

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
