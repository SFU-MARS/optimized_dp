import numpy as np
from scipy.integrate import solve_ivp


class DubinsCar:
    def __init__(self, x=[0, 0, 0], wMax=1, speed=1, dMax=[0, 0, 0], uMode="min", dMode="max"):
        self.x = x
        self.wMax = wMax
        self.speed = speed
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = self.wMax
        # Just create and pass back, even though they're not used
        if spat_deriv[2] > 0:
            if self.uMode == "min":
                opt_w = -opt_w
        elif spat_deriv[2] < 0:
            if self.uMode == "max":
                opt_w = -opt_w
        return opt_w

    def dynamics(self, t, state, uOpt):
        x_dot = self.speed * np.cos(state[2])
        y_dot = self.speed * np.sin(state[2])
        theta_dot = uOpt
        print(x_dot, y_dot, theta_dot)

        return x_dot, y_dot, theta_dot


def update_state(dyn_sys, control, delta_t):
    """
    Simulate apply control to dynamical systems for delta_t time

    Args:
        dyn_sys: dynamic system
        control: control
        delta_t: duration of control

    Returns:

    """
    init_state = dyn_sys.x
    t_span = [0, delta_t]
    solution = solve_ivp(dyn_sys.dynamics, t_span, init_state, args=[control], dense_output=True)
    final_state = solution.y[:, -1]
    print(solution.t)
    print(solution.y)
    print(final_state)
    dyn_sys.x = final_state

if __name__ in "__main__":
    dyn_sys = DubinsCar()
    control = 0
    # TODO: this should be a base method
    update_state(dyn_sys, control, 1)

