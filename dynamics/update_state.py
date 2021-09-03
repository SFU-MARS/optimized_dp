import numpy as np
from scipy.integrate import solve_ivp


def update_state(dyn_sys, u, d, delta_t):
    """
    Simulate apply control to dynamical systems for delta_t time

    Args:
        dyn_sys: dynamic system
        u: control
        d: disturbance
        delta_t: duration of control

    Returns:

    """
    init_state = dyn_sys.x
    t_span = [0, delta_t]
    solution = solve_ivp(dyn_sys.dynamics_non_hcl, t_span, init_state, args=[u, d], dense_output=True)
    final_state = solution.y[:, -1]
    dyn_sys.x = final_state

    # TODO account for periodic bounds
    dyn_sys.update_state()