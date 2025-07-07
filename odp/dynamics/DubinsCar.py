import heterocl as hcl
import numpy as np

class DubinsCar:
    def __init__(self, x=[0,0,0], wMax=1, speed=1, dMax=[0,0,0], uMode="min", dMode="max"):
        self.x = x
        self.wMax = wMax
        self.speed = speed
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.wMax, "opt_w")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        with hcl.if_(spat_deriv[2] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w
        with hcl.elif_(spat_deriv[2] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w
        return (opt_w[0], in3[0], in4[0])

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
        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = self.speed*hcl.cos(state[2])
        y_dot[0] = self.speed*hcl.sin(state[2])
        theta_dot[0] = uOpt[0]

        return (x_dot[0], y_dot[0], theta_dot[0])
    
    def optCtrl_inPython(self, spat_deriv):
        opt_u = self.wMax
        if spat_deriv[2] > 0:
            if self.uMode == "min":
                opt_u = - self.wMax
        else:
            if self.uMode == "max":
                opt_u = - self.wMax
        
        return np.array(opt_u)
    
    def dynamics_inPython(self, state, action):
        """Return the partial derivative equations of one agent.

        Args:
            state (np.ndarray, shape(3, )): the state of one agent
            action (np.ndarray, shape (1, )): the action of one agent
        """
        dx = self.speed * np.cos(state[2])
        dy = self.speed * np.sin(state[2])
        dtheta = action[0]
        return (dx, dy, dtheta)
    
    def forward(self, ctrl_freq, current_state, u):
        # Forward the dubincar dynamics with one step
        x, y, theta = current_state
        dt = 1.0 / ctrl_freq
        
        # Forward-Euler method
        next_x = x + self.speed * np.cos(theta) * dt
        next_y = y + self.speed * np.sin(theta) * dt
        next_theta_raw = theta + u * dt

        def check_theta(angle):
            # Make sure the angle is in the range of [0, 2*pi)
            while angle >=2*np.pi:
                angle -= 2 * np.pi
            while angle < 0:
                angle += 2 * np.pi

            return angle

        # Check the boundary
        next_theta = check_theta(next_theta_raw)
        next_state = (next_x, next_y, next_theta)
        
        return next_state
        
        