import heterocl as hcl
import numpy as np

""" 6D DUBINS AIRPLANE DYNAMICS IMPLEMENTATION 
 x_dot = v * cos(gamma) * cos(psi) + d1
 y_dot = v * cos(gamma) * sin(psi) + d2
 z_dot = v * sin(gamma) +d3
 psi_dot = g/v * tan(phi)
 gamma_dot = u_gamma
 phi_dot = u_phi
 """

class DubinsAirplane6D:
    def __init__(self, x=[0,0,0,0,0,0], uMin=[-1,-1], uMax=[1,1], dMin=[0,0,0],
                 dMax=[0,0,0], uMode="min", dMode="max", v=1.0):
        """Creates a Dubins Airplane with the following states:
           X position, Y position, Z position (altitude), heading (psi), flight path angle (gamma), roll angle (phi)
           The controls are the flight path angle rate (u_gamma), roll angle rate (u_phi), and acceleration (a).
           The disturbances are the noise in the velocity components.
        Args:
            x (list, optional): Initial state. Defaults to [0,0,0,0,0,0].
            uMin (list, optional): Lower bound of user control. Defaults to [-1,-1,-1].
            uMax (list, optional): Upper bound of user control. Defaults to [1,1,1].
            dMin (list, optional): Lower bound of disturbance to user control. Defaults to [-0.25,-0.25,-0.25].
            dMax (list, optional): Upper bound of disturbance to user control. Defaults to [0.25,0.25,0.25].
            uMode (str, optional): Accepts either "min" or "max".
                                   * "min" : have optimal control reach goal
                                   * "max" : have optimal control avoid goal
                                   Defaults to "min".
            dMode (str, optional): Accepts whether "min" or "max" and should be opposite of uMode.
                                   Defaults to "max".
            g (float, optional): Gravitational acceleration. Defaults to 9.81.
        """
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        self.g = 9.81
        self.v = v
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
        # x_dot     = v * cos(gamma) * cos(psi) + d_1
        # y_dot     = v * cos(gamma) * sin(psi) + d_2
        # z_dot     = v * sin(gamma) + d_3
        # psi_dot   = g/v * tan(phi)
        # gamma_dot = u_gamma
        # phi_dot   = u_phi

        # Graph takes in 6 possible inputs, by default, for now
        opt_u_gamma = hcl.scalar(self.uMax[0], "u_gamma")
        opt_u_phi = hcl.scalar(self.uMax[1], "u_phi")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        in5 = hcl.scalar(0, "in5")
        in6 = hcl.scalar(0, "in6")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[4] > 0):
                opt_u_gamma[0] = self.uMin[0]
            with hcl.if_(spat_deriv[5] > 0):
                opt_u_phi[0] = self.uMin[1]
        else:
            with hcl.if_(spat_deriv[4] < 0):
                opt_u_gamma[0] = self.uMin[0]
            with hcl.if_(spat_deriv[5] < 0):
                opt_u_phi[0] = self.uMin[1]
        # return 4, 5, 6 even if you don't use them
        return (opt_u_gamma[0], opt_u_phi[0], in3[0], in4[0], in5[0], in6[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 6 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        # Just create and pass back, even though they're not used
        d4 = hcl.scalar(0, "d4")
        d5 = hcl.scalar(0, "d5")
        d6 = hcl.scalar(0, "d6")

        if self.dMode == "max":
            with hcl.if_(spat_deriv[0] > 0):
                d1[0] = self.dMax[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMin[0]
            with hcl.if_(spat_deriv[1] > 0):
                d2[0] = self.dMax[1]
            with hcl.elif_(spat_deriv[1] < 0):
                d2[0] = self.dMin[1]
            with hcl.if_(spat_deriv[2] > 0):
                d3[0] = self.dMax[2]
            with hcl.elif_(spat_deriv[2] < 0):
                d3[0] = self.dMin[2]
        else:
            with hcl.if_(spat_deriv[0] > 0):
                d1[0] = self.dMin[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMax[0]
            with hcl.if_(spat_deriv[1] > 0):
                d2[0] = self.dMin[1]
            with hcl.elif_(spat_deriv[1] < 0):
                d2[0] = self.dMax[1]
            with hcl.if_(spat_deriv[2] > 0):
                d3[0] = self.dMin[2]
            with hcl.elif_(spat_deriv[2] < 0):
                d3[0] = self.dMax[2]

        return (d1[0], d2[0], d3[0], d4[0], d5[0], d6[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        z_dot = hcl.scalar(0, "z_dot")
        psi_dot = hcl.scalar(0, "psi_dot")
        gamma_dot = hcl.scalar(0, "gamma_dot")
        phi_dot = hcl.scalar(0, "phi_dot")

        x_dot[0] = self.v * hcl.cos(state[4]) * hcl.cos(state[3]) + dOpt[0]
        y_dot[0] = self.v * hcl.cos(state[4]) * hcl.sin(state[3]) + dOpt[1]
        z_dot[0] = self.v * hcl.sin(state[4]) + dOpt[2]
        psi_dot[0] = self.g / self.v * (hcl.sin(state[5]) / hcl.cos(state[5]))
        gamma_dot[0] = uOpt[1]
        phi_dot[0] = uOpt[2]

        return (x_dot[0], y_dot[0], z_dot[0], psi_dot[0], gamma_dot[0], phi_dot[0])

    def optCtrl_inPython(self, spat_deriv):
        opt_u_gamma = self.uMax[0]
        opt_u_phi = self.uMax[1]
        if self.uMode == "min":
            if spat_deriv[4] > 0:
                opt_u_gamma = self.uMin[0]
            if spat_deriv[5] > 0:
                opt_u_phi = self.uMin[2]
        else:
            if spat_deriv[4] < 0:
                opt_u_gamma = -self.uMin[0]
            if spat_deriv[5] < 0:
                opt_u_phi = -self.uMin[2]

        return opt_u_gamma, opt_u_phi

    def dynamics_inPython(self, state, action):
        """Compute the first-order derivative of one agent. No disturbance now.

        Args:
            state (np.ndarray, shape(6, )): the state of one agent
            action (np.ndarray, shape (3, )): the action of one agent
        Return:
            a tuple of the first-order derivative of the dynamics
        """
        x_dot = self.v * np.cos(state[4]) * np.cos(state[3])
        y_dot = self.v * np.cos(state[4]) * np.sin(state[3])
        z_dot = self.v * np.sin(state[4])
        psi_dot = self.g / self.v * np.tan(state[5])
        gamma_dot = action[1]
        phi_dot = action[2]

        return (x_dot, y_dot, z_dot, psi_dot, gamma_dot, phi_dot)

    def forward(self, ctrl_freq, current_state, action):
        """Compute the next state of the agent, no disturbance is considered now.

        Args:
            ctrl_freq (int): the control frequency
            current_state (np.ndarray, shape(6, )): the state of one agent
            action (np.ndarray, shape (3, )): the action of one agent
        Return:
            next_state (tuple, len 6): the next state of the agent
        """
        # Forward the Dubins airplane dynamics with one step
        x, y, z, psi, gamma, phi = current_state
        dt = 1.0 / ctrl_freq

        # Forward-Euler method
        next_x = x + self.v * np.cos(gamma) * np.cos(psi) * dt
        next_y = y + self.v * np.cos(gamma) * np.sin(psi) * dt
        next_z = z + self.v * np.sin(gamma) * dt
        next_psi = psi + (self.g / self.v) * np.tan(phi) * dt
        next_gamma = gamma + action[1] * dt
        next_phi = phi + action[2] * dt

        def check_angle(angle):
            # Make sure the angle is in the range of [0, 2*pi)
            while angle >= 2 * np.pi:
                angle -= 2 * np.pi
            while angle < 0:
                angle += 2 * np.pi
            return angle

        # Check the boundary for angles
        next_psi = check_angle(next_psi)
        next_gamma = check_angle(next_gamma)
        next_phi = check_angle(next_phi)

        next_state = (next_x, next_y, next_z, next_psi, next_gamma, next_phi)

        return next_state