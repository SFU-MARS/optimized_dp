import heterocl as hcl
import numpy as np

""" 3D DUBINS CAR DYNAMICS with 2 CONTROL INPUTS IMPLEMENTATION 
 x_dot = v * cos(theta)
 y_dot = v * sin(theta)
 theta_dot = w
 u[0] = speed
 u[1] = w
 """


class DubinsCar2:
    def __init__(self, x=[0,0,0], uMin=[-0.1, -1.0], uMax =[0.8, 1.0], dMax=[0,0,0], uMode="min", dMode="max"):
        self.x = x
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode
        self.speedMin = uMin[0]
        self.speedMax = uMax[0]
        self.wMin = uMin[1]
        self.wMax = uMax[1]

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.wMax, "opt_w")
        opt_speed = hcl.scalar(self.speedMax, "opt_speed")
        # Just create and pass back, even though they're not used
        in4 = hcl.scalar(0, "in4")
        # with hcl.if_(spat_deriv[2] > 0):
        #     with hcl.if_(self.uMode == "min"):
        #         opt_w[0] = -opt_w
        # with hcl.elif_(spat_deriv[2] < 0):
        #     with hcl.if_(self.uMode == "max"):
        #         opt_w[0] = -opt_w
        coefficient = spat_deriv[0]*np.cos(state[2]) + spat_deriv[1]*np.sin(state[2])
        with hcl.if_(self.uMode == "min"):
            with hcl._if(coefficient > 0):
                opt_speed[0] = self.speedMin
            with hcl.if_(spat_deriv[2] > 0):
                opt_w[0] = self.wMin
        with hcl.if_(self.uMode == "max"):
            with hcl._if(coefficient < 0):
                opt_speed[0] = self.speedMin
            with hcl.elif_(spat_deriv[2] < 0):
                opt_w[0] = self.wMin
            
        return (opt_w[0], opt_speed[0], in4[0])

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

        x_dot[0] = uOpt[0]*hcl.cos(state[2])
        y_dot[0] = uOpt[0]*hcl.sin(state[2])
        theta_dot[0] = uOpt[1]

        return (x_dot[0], y_dot[0], theta_dot[0])
    
    def optCtrl_inPython(self, state, spat_deriv):
        opt_w = self.wMax
        opt_speed = self.speedMax
        coefficient = spat_deriv[0]*np.cos(state[2]) + spat_deriv[1]*np.sin(state[2])
        
        if self.uMode == "min":
            if spat_deriv[2] > 0:
                opt_w = self.wMin
            if coefficient > 0:
                opt_speed = self.speedMin
        else:
            if spat_deriv[2] < 0:
                opt_w = self.wMin
            if coefficient < 0:
                opt_speed = self.speedMin
        # if spat_deriv[2] > 0:
        #     if self.uMode == "min":
        #         opt_w = - self.wMax
        # else:
        #     if self.uMode == "max":
        #         opt_w = - self.wMax
        
        return np.array(opt_w, opt_speed)
    
    def dynamics_inPython(self, state, control):
        """Return the partial derivative equations of one agent.

        Args:
            state (np.ndarray, shape(3, )): the state of one agent
            action (np.ndarray, shape (1, )): the action of one agent
        """
        dx = control[0] * np.cos(state[2])
        dy = control[0] * np.sin(state[2])
        dtheta = control[1]
        return (dx, dy, dtheta)
    
    def forward(self, ctrl_freq, current_state, control):
        # Forward the dubincar dynamics with one step
        x, y, theta = current_state
        dt = 1.0 / ctrl_freq
        
        # Forward-Euler method
        next_x = x + control[0] * np.cos(theta) * dt
        next_y = y + control[0] * np.sin(theta) * dt
        next_theta_raw = theta + control[1] * dt

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
        
        