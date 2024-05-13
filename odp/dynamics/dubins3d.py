import heterocl as hcl
import numpy as np

from .base import DynamicsBase

__all__ = ['DubinsCar', 'DubinsCapture']


class DubinsCar3D(DynamicsBase):

    state_dims = 3
    ctrl_dims = 1
    dstb_dims = 0

    X, Y, THETA = range(state_dims)
    TURN, = range(ctrl_dims)

    def __init__(self, mode='reach', *, speed=1., max_turn_rate=1.):
        super().__init__(ctrl_range=[[-max_turn_rate], 
                                     [+max_turn_rate]], 
                         dstb_range=[[], []],
                         mode=mode)

        self.speed = speed
        self.max_turn_rate = max_turn_rate

    def dynamics(self, dx, t, x, u, d):

        # x_dot = v * cos(theta)
        dx[self.X] = self.speed * hcl.cos(x[self.THETA])

        # y_dot = v * sin(theta)
        dx[self.Y] = self.speed * hcl.sin(y[self.THETA])

        # theta_dot = u
        dx[self.THETA] = u[self.TURN]

    def opt_ctrl(self, u, dv, t, x):

        uMin, uMax = self.ctrl_range

        if self.mode['uMode'] == 'max':
            with hcl.if_(0 <= dv[self.THETA]):
                u[self.TURN] = uMax[self.TURN]
            with hcl.else_():
                u[self.TURN] = uMin[self.TURN]
        else:
            with hcl.if_(dv[self.THETA] < 0):
                u[self.TURN] = uMax[self.TURN]
            with hcl.else_():
                u[self.TURN] = uMax[self.TURN]

    def opt_dstb(self, d, dv, t, x):
        pass

class DubinsCapture(DynamicsBase):

    state_dims = 3
    ctrl_dims = 1
    dstb_dims = 2

    def __init__(self, mode='reach', *, speed=1., max_turn_rate=1., max_disturbance=1.):
        super().__init__(ctrl_range=[[-max_turn_rate], 
                                     [+max_turn_rate]], 
                         dstb_range=[[-max_disturbance] * self.dstb_dims, 
                                     [+max_disturbance] * self.dstb_dims],
                         mode=mode)

        self.speed = speed
        self.max_turn_rate = max_turn_rate
        self.max_disturbance = max_disturbance

    def dynamics(self, dx, t, x, u, d):
        dx[0] = -self.speed + self.speed*hcl.cos(x[2]) + u[0]*x[1]
        dx[1] =             + self.speed*hcl.sin(x[2]) - u[0]*x[0]
        dx[2] = d[0] - u[0]

    def opt_ctrl(self, u, dv, t, x):

        uMin, uMax = self.ctrl_range

        # Declare a variable
        a_term = hcl.scalar(0, "a_term")
        a_term.v = dv[0]*x[1] - dv[1]*x[0] - dv[2]

        u[0] = uMax[0]

        if self.mode['uMode'] == 'max':
            with hcl.if_(0 <= a_term.v):
                u[0] = uMax[0]
            with hcl.else_():
                u[0] = uMin[0]
        else:
            with hcl.if_(a_term.v < 0):
                u[0] = uMax[0]
            with hcl.else_():
                u[0] = uMin[0]
    
    def opt_dstb(self, d, dv, t, x):
        
        dMin, dMax = self.dstb_range

        # Declare a variable
        b_term = hcl.scalar(0, "b_term")
        b_term.v = dv[2]

        if self.mode['dMode'] == 'max':
            with hcl.if_(0 <= b_term.v):
                d[0] = dMax[0]
            with hcl.else_():
                d[0] = dMin[0]
        else:
            with hcl.if_(b_term.v < 0):
                d[0] = dMax[0]
            with hcl.else_():
                d[0] = dMin[0]
