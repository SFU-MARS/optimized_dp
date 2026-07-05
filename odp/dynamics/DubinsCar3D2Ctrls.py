import heterocl as hcl
import numpy as np

"""3D Dubins car with two controls, aligned with ToyDubinsCarNavEnv (odp/dynamics/dubins_car.py).

Continuous dynamics:
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = w

Physical controls (reachability / forward):
    v in [0, v_max],  w in [-w_max, w_max]

Gym normalized actions a in [-1, 1]^2:
    v = (a[0] + 1) / 2 * v_max
    w = a[1] * w_max
"""


class DubinsCar3D2Ctrls:
    def __init__(
        self,
        x=(0, 0, 0),
        uMin=None,
        uMax=None,
        v_max=1.0,
        w_max=1.0,
        dt=0.1,
        dMax=(0, 0, 0),
        uMode="min",
        dMode="max",
    ):
        self.x = x
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.dt = float(dt)

        if uMin is None:
            uMin = [0.0, -self.w_max]
        if uMax is None:
            uMax = [self.v_max, self.w_max]

        self.speedMin = uMin[0]
        self.speedMax = uMax[0]
        self.wMin = uMin[1]
        self.wMax = uMax[1]

    @staticmethod
    def wrap_theta(theta):
        """Wrap heading to [-pi, pi], matching ToyDubinsCarNavEnv."""
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def action_to_control(self, action):
        """Map Gym actions in [-1, 1]^2 to physical (v, w)."""
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        v = (action[..., 0] + 1.0) / 2.0 * self.v_max
        w = action[..., 1] * self.w_max
        return v, w

    def control_to_action(self, v, w):
        """Map physical (v, w) to Gym actions in [-1, 1]^2."""
        v = np.asarray(v, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        a0 = 2.0 * v / self.v_max - 1.0
        a1 = w / self.w_max
        return np.stack([np.clip(a0, -1.0, 1.0), np.clip(a1, -1.0, 1.0)], axis=-1)

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.wMax, "opt_w")
        opt_speed = hcl.scalar(self.speedMax, "opt_speed")
        # Just create and pass back, even though they're not used
        in4 = hcl.scalar(0, "in4")
        # Declare hcl scalars for the coefficient
        deriv0 = hcl.scalar(0, "deriv0")
        deriv1 = hcl.scalar(0, "deriv1")
        theta = hcl.scalar(0, "theta")
        deriv0[0] = spat_deriv[0]
        deriv1[0] = spat_deriv[1]
        theta[0] = state[2]
        coefficient = deriv0[0] * hcl.cos(theta[0]) + deriv1[0] * hcl.sin(theta[0])

        with hcl.if_(self.uMode == "min"):
            with hcl.if_(coefficient > 0):
                opt_speed[0] = self.speedMin
            with hcl.if_(spat_deriv[2] > 0):
                opt_w[0] = self.wMin
        with hcl.if_(self.uMode == "max"):
            with hcl.if_(coefficient < 0):
                opt_speed[0] = self.speedMin
            with hcl.elif_(spat_deriv[2] < 0):
                opt_w[0] = self.wMin

        return (opt_speed[0], opt_w[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = uOpt[0] * hcl.cos(state[2])
        y_dot[0] = uOpt[0] * hcl.sin(state[2])
        theta_dot[0] = uOpt[1]

        return (x_dot[0], y_dot[0], theta_dot[0])

    def optCtrl_inPython(self, state, spat_deriv):
        opt_w = self.wMax
        opt_speed = self.speedMax
        coefficient = spat_deriv[0] * np.cos(state[2]) + spat_deriv[1] * np.sin(state[2])

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
        return np.array([opt_speed, opt_w])

    def dynamics_inPython(self, state, control):
        """Return time derivatives (dx, dy, dtheta) for physical control (v, w)."""
        dx = control[0] * np.cos(state[2])
        dy = control[0] * np.sin(state[2])
        dtheta = control[1]
        return (dx, dy, dtheta)

    def forward(self, *args, dt=None, ctrl_freq=None):
        """Euler step with physical control (v, w).

        Preferred: forward(current_state, control, dt=None)
        Legacy:    forward(ctrl_freq, current_state, control)
        """
        if len(args) == 3 and isinstance(args[0], (int, float)):
            ctrl_freq, current_state, control = args
            dt = 1.0 / float(ctrl_freq)
        elif len(args) == 2:
            current_state, control = args
            if dt is None:
                if ctrl_freq is not None:
                    dt = 1.0 / float(ctrl_freq)
                else:
                    dt = self.dt
        else:
            raise TypeError(
                "forward() expects (current_state, control) or "
                "(ctrl_freq, current_state, control)"
            )

        x, y, theta = current_state
        v, w = control[0], control[1]

        next_x = x + v * np.cos(theta) * dt
        next_y = y + v * np.sin(theta) * dt
        next_theta = self.wrap_theta(theta + w * dt)
        return (next_x, next_y, next_theta)

    def forward_from_action(self, current_state, action, dt=None):
        """Euler step using Gym-normalized actions in [-1, 1]^2."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        v, w = self.action_to_control(action)
        return self.forward(current_state, (float(v), float(w)), dt=dt)
