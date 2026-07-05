import heterocl as hcl

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
    # def opt_dstb(self, spat_deriv):
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
    
    def optCtrl_inPython(self, state, spat_deriv):
        opt_a = self.uMax[0]
        opt_w = self.uMax[1]
        
        if self.uMode == "min":
            if spat_deriv[2] >= 0.0:
                opt_a = self.uMin[0]
            if spat_deriv[3] >= 0.0:
                opt_w = self.uMin[1]
        else:  # "max"
            if spat_deriv[2] < 0.0:
                opt_a = self.uMin[0]
            if spat_deriv[3] < 0.0:
                opt_w = self.uMin[1]
                
        return (opt_a, opt_w)
    
    def optDistb_inPython(self, state, spat_deriv):
        distb1 = self.dMax[0]
        distb2 = self.dMax[1]
        
        if self.dMode == "min":
            if spat_deriv[0] >= 0.0:
                distb1 = self.dMin[0]
            if spat_deriv[1] >= 0.0:
                distb2 = self.dMin[1]
        else:  # "max"
            if spat_deriv[0] < 0.0:
                distb1 = self.dMin[0]
            if spat_deriv[1] < 0.0:
                distb2 = self.dMin[1]
        
        return (dist1, distb2)
        
    
    def dynamics_inPython(self, state, control, disturbance):
        dx = state[2] * np.cos(state[3]) + disturbance[0]
        dy = state[2] * np.sin(state[3]) + disturbance[1]
        dv = control[0]
        dtheta = control[1]

        return (dx, dy, dv, dtheta)
    
    
    def forward(self, ctrl_freq, current_state, control, disturbance):
        x, y, v, theta = current_state
        dt = 1.0 / ctrl_freq
        dx, dy, dv, dtheta = self.dynamics_inPython(current_state, control, disturbance)
        
        # Forward-Euler method
        next_x = x + dx * dt
        next_y = y + dy * dt
        next_v = v + dv * dt
        next_theta_raw = theta + dtheta * dt
        
        def check_theta(angle):
            # Make sure the angle is in the range of [-pi, pi)
            while angle >=np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi

            return angle
        

        # Check the boundary
        next_theta = check_theta(next_theta_raw)

        return (next_x, next_y, next_v, next_theta)
        
        