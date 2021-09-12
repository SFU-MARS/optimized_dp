from Grid.GridProcessing import Grid
from spatialDerivatives.first_order_generic import spa_deriv
from dynamics.update_state import update_state
import numpy as np


def compute_opt_traj(grid: Grid, V, tau, dyn_sys, subsamples=1):
    """
    Computes the optimal trajectory, controls and disturbance to a minimal BRT/BRS

    Args:
        grid:
        V:
        tau:
        dyn_sys:
        subsamples: Number of times opt_u and opt_d are calculated within dt

    Returns:
        traj: State of dyn_sys from time tau[-1] to tau[0]
        opt_u: Optimal control at each time
        opt_d: Optimal disturbance at each time

    """
    assert V.shape[-1] == len(tau)
    if grid.get_value(V[..., -1], dyn_sys.x) > 0:
        raise ValueError(f"Value must be within BRT/BRS for optimal trajectory to be computed")

    dt = tau[1] - tau[0] / subsamples

    # first entry is dyn_sys at time tau[-1]
    # second entry is dyn_sys at time tau[-2]...
    traj = np.empty((V.shape[-1], len(dyn_sys.x)))
    traj[0] = dyn_sys.x

    opt_u = []
    opt_d = []

    t_earliest = -1

    for time_idx, time in enumerate(tau, start=1):
        brs_at_t = V[..., t_earliest]

        gradient = spa_deriv(grid.get_index(dyn_sys.x), brs_at_t, grid, periodic_dims=[2])

        for _ in range(subsamples):
            u = dyn_sys.opt_ctrl_non_hcl(_, dyn_sys.x, gradient)
            d = dyn_sys.opt_dstb_non_hcl(_, dyn_sys.x, gradient)
            update_state(dyn_sys, u, d, dt)
            opt_u.append(u)
            opt_d.append(d)

        if time_idx != V.shape[-1]:
            traj[time_idx] = dyn_sys.x

    return traj, opt_u, opt_d

