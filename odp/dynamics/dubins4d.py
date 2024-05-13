import heterocl as hcl
import numpy as np

from .base import DynamicsBase

__all__ = ['DubinsCar4D', 'DubinsCar4D2']


class DubinsCar4D(DynamicsBase):
    """ 4D DUBINS CAR DYNAMICS IMPLEMENTATION 
    x_dot = v * cos(theta) + d_1
    y_dot = v * sin(theta) + d_2
    v_dot = a
    theta_dot = w
    """

    state_dims = 4
    ctrl_dims = 2
    dstb_dims = 2

    X, Y, V, THETA = range(state_dims)
    ACC, TURN = range(ctrl_dims)

    def __init__(self, 
                 ctrl_range=[[-1, -1], 
                             [+1, +1]], 
                 dstb_range=[[-0.25, -0.25],
                             [+0.25, +0.25]], 
                 mode='reach'):
        super().__init__(ctrl_range, dstb_range, mode)

    def dynamics(self, dx, t, x, u, d):
        # x_dot = v * cos(theta) + d_1
        dx[self.X] = (x[self.V] * hcl.cos(x[self.THETA]) 
                      + d[self.X])

        # y_dot = v * sin(theta) + d_2
        dx[self.Y] = (x[self.V] * hcl.sin(x[self.THETA])
                      + d[self.Y])

        # v_dot = a
        dx[self.V] = u[self.ACC]

        # theta_dot = w
        dx[self.THETA] = u[self.TURN]

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        v_dot = hcl.scalar(0, "v_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = state[2] * hcl.cos(state[3]) + dOpt[0]
        y_dot[0] = state[2] * hcl.sin(state[3]) + dOpt[1]
        v_dot[0] = uOpt[0]
        theta_dot[0] = uOpt[1]

        return (x_dot[0], y_dot[0], v_dot[0] ,theta_dot[0])


    def opt_ctrl(self, u, dv, t, x):
        
        uMin, uMax = self.ctrl_range

        if self.mode['uMode'] == "max":
            # Acceleration
            with hcl.if_(dv[self.THETA] < 0):
                u[self.ACC] = uMin[self.ACC]
            with hcl.else_():
                u[self.ACC] = uMax[self.ACC]
            # Turn rate
            with hcl.if_(dv[self.THETA] < 0):
                u[self.TURN] = uMin[self.TURN]
            with hcl.else_():
                u[self.TURN] = uMax[self.TURN]
        else:
            # Acceleration
            with hcl.if_(dv[self.V] > 0):
                u[self.ACC] = uMin[self.ACC]
            with hcl.else_():
                u[self.ACC] = uMax[self.ACC]
            # Turn rate
            with hcl.if_(dv[self.THETA] > 0):
                u[self.TURN] = uMin[self.TURN]
            with hcl.else_():
                u[self.TURN] = uMax[self.TURN]


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


class DubinsCar4D2(DynamicsBase):
    """
    4D DUBINS CAR DYNAMICS IMPLEMENTATION
    used to model dynamics of Jetracer

    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    v_dot = a
    theta_dot = v * tan(delta) / L

    delta := steering angle
    L := wheelbase of car

    (6.2) https://arxiv.org/pdf/1711.03449.pdf
    """

    state_dims = 4
    ctrl_dims = 2
    dstb_dims = 2

    X, Y, V, THETA = range(state_dims)
    ACC, TURN = range(ctrl_dims)

    def __init__(self,
                 ctrl_range=[[-1.5, -np.pi/18],
                             [+1.5, +np.pi/18]],
                 dstb_range=[[-0.0, -0.0],
                             [+0.0, +0.0]],
                 mode='reach',
                 *,
                 wheelbase=0.3): # wheelbase of Tamiya TT02
        super().__init__(ctrl_range, dstb_range, mode)

        self.wheelbase = wheelbase

    def dynamics(self, dx, t, x, u, d):
        # x_dot = v * cos(theta)
        dx[self.X] = (x[self.V] * hcl.cos(x[self.THETA]) + d[self.X])

        # y_dot = v * sin(theta)
        dx[self.Y] = (x[self.V] * hcl.sin(x[self.THETA]) + d[self.Y])

        # v_dot = a
        dx[self.V] = (u[self.ACC] + d[self.V])

        # theta_dot = v * tan(u1) / L
        hcl_tan = lambda a: hcl.sin(a) / hcl.cos(a)
        dx[self.THETA] = (x[self.V] * hcl_tan(u[self.TURN]) / self.wheelbase + d[self.THETA])

    def opt_ctrl(self, u, dv, t, x):
            
        uMin, uMax = self.ctrl_range

        if self.mode['uMode'] == "max":
            # Acceleration
            with hcl.if_(0 <= dv[self.V]):
                u[self.ACC] = uMax[self.ACC]
            with hcl.else_():
                u[self.ACC] = uMin[self.ACC]
            # Turn rate
            with hcl.if_(0 <= dv[self.THETA]):
                u[self.TURN] = uMax[self.TURN]
            with hcl.else_():
                u[self.TURN] = uMin[self.TURN]
        else:
            # Acceleration
            with hcl.if_(0 <= dv[self.V]):
                u[self.ACC] = uMin[self.ACC]
            with hcl.else_():
                u[self.ACC] = uMax[self.ACC]
            # Turn rate
            with hcl.if_(0 <= dv[self.THETA]):
                u[self.TURN] = uMin[self.TURN]
            with hcl.else_():
                u[self.TURN] = uMax[self.TURN]

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

