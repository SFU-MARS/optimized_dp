import heterocl as hcl
import numpy as np

""" 4D DUBINS CAR DYNAMICS IMPLEMENTATION 
 x_dot = v * cos(theta) + d_1
 y_dot = v * sin(theta) + d_2
 v_dot = a
 theta_dot = w
 """
class DubinsCar4D:
    def __init__(self, x=[0,0,0,0], uMin = [-1,-1], uMax = [1,1], dMin = [-0.25,-0.25],
                 dMax=[0.25,0.25], uMode="min", dMode="max"):
        """Creates a Dublin Car with the following states:
           X position, Y position, speed, heading
           The controls are the acceleration and turn rate (angular speed)
           The disturbances are the noise in the velocity components.
        Args:
            x (list, optional): Initial state . Defaults to [0,0,0,0].
            uMin (list, optional): Lowerbound of user control. Defaults to [-1,-1].
            uMax (list, optional): Upperbound of user control.
                                   Defaults to [1,1].
            dMin (list, optional): Lowerbound of disturbance to user control, . Defaults to [-0.25,-0.25].
            dMax (list, optional): Upperbound of disturbance to user control. Defaults to [0.25,0.25].
            uMode (str, optional): Accepts either "min" or "max".
                                   * "min" : have optimal control reach goal
                                   * "max" : have optimal control avoid goal
                                   Defaults to "min".
            dMode (str, optional): Accepts whether "min" or "max" and should be opposite of uMode.
                                   Defaults to "max".
        """
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        assert(uMode in ["min", "max"])
        self.uMode = uMode
        if uMode == "min":
            assert(dMode == "max")
        else:
            assert(dMode == "min")
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # System dynamics
        # x_dot     = v * cos(theta) + d_1
        # y_dot     = v * sin(theta) + d_2
        # v_dot     = a
        # theta_dot = w

        # Graph takes in 4 possible inputs, by default, for now
        opt_a = hcl.scalar(self.uMax[0], "opt_a")
        opt_w = hcl.scalar(self.uMax[1], "opt_w")
        # Just create and pass back, even though they're not used
        in3   = hcl.scalar(0, "in3")
        in4   = hcl.scalar(0, "in4")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[2] > 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(spat_deriv[3] > 0):
                opt_w[0] = self.uMin[1]
        else:
            with hcl.if_(spat_deriv[2] < 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(spat_deriv[3] < 0):
                opt_w[0] = self.uMin[1]
        # return 3, 4 even if you don't use them
        return (opt_a[0] ,opt_w[0], in3[0], in4[0])

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
        d4 = hcl.scalar(0, "d4")

        #with hcl.if_(self.dMode == "max"):
        if self.dMode == "max":
            with hcl.if_(spat_deriv[0] > 0):
                d1[0] = self.dMax[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMin[0]
            with hcl.if_(spat_deriv[1] > 0):
                d2[0] = self.dMax[1]
            with hcl.elif_(spat_deriv[1] < 0):
                d2[0] = self.dMin[1]
        else:
            with hcl.if_(spat_deriv[0] > 0):
                d1[0] = self.dMin[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMax[0]
            with hcl.if_(spat_deriv[1] > 0):
                d2[0] = self.dMin[1]
            with hcl.elif_(spat_deriv[1] < 0):
                d2[0] = self.dMax[1]

        return (d1[0], d2[0], d3[0], d4[0])

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
    
    def optCtrl_inPython(self, spat_deriv):
        opt_a = self.uMax[0]
        opt_w = self.uMax[1]
        if self.uMode == "min":
            if spat_deriv[2] > 0:
                opt_a = self.uMin[0]
            if spat_deriv[3] > 0:
                opt_w = self.uMin[1]
        else:
            if spat_deriv[2] < 0:
                opt_a = - self.uMin[0]
            if spat_deriv[3] < 0:
                opt_w = - self.uMin[1]
        
        return opt_a, opt_w

    def dynamics_inPython(self, state, action):
        """Compute the first-order derivative of one agent. No distburbance now.

        Args:
            state (np.ndarray, shape(4, )): the state of one agent
            action (np.ndarray, shape (2, )): the action of one agent
        Return:
            a tuple of the first-order derivative of the dynamics
        """
        x_dot = state[2] * np.cos(state[3])
        y_dot = state[2] * np.sin(state[3])
        v_dot = action[0]
        theta_dot = action[1]
        return (x_dot, y_dot, v_dot, theta_dot)
    
    def forward(self, ctrl_freq, current_state, action):
        """Compute the next state of the agent, no disturbance is considered now.

        Args:
            ctrl_freq (int): the control frequency
            current_state (np.ndarray, shape(4, )): the state of one agent
            action (np.ndarray, shape (2, )): the action of one agent
        Return:
            next_state (tuple, len 4): the next state of the agent
        """
        # Forward the dubincar dynamics with one step
        x, y, v, theta = current_state
        dt = 1.0 / ctrl_freq

        # Forward-Euler method
        next_x = x + current_state[2] * np.cos(theta) * dt
        next_y = y + current_state[2] * np.sin(theta) * dt
        next_v = v + action[0] * dt
        next_theta_raw = theta + action[1] * dt

        def check_theta(angle):
            # Make sure the angle is in the range of [0, 2*pi)
            while angle >=2*np.pi:
                angle -= 2 * np.pi
            while angle < 0:
                angle += 2 * np.pi

            return angle

        # Check the boundary
        next_theta = check_theta(next_theta_raw)
        next_state = (next_x, next_y, next_v, next_theta)
        
        return next_state