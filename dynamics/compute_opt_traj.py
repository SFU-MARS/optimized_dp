from Grid.GridProcessing import Grid
from spatialDerivatives.first_order_generic import spa_deriv
from dynamics.update_state import update_state
import numpy as np


def compute_opt_traj(grid: Grid, V, tau, dyn_sys):
    assert V.shape[-1] == len(tau)

    subsamples = 1
    dt = tau[1] - tau[0] / subsamples

    # traj stores state of dyn_sys from look_back_length -> 0
    # first entry is dyn_sys at time -t
    # second entry is dyn_sys at time -t + delta t
    # traj(t, x)
    traj = np.empty((V.shape[-1], len(dyn_sys.x)))
    opt_u = []
    opt_d = []

    traj[0] = dyn_sys.x
    t_earliest = 0

    for time_idx, time in enumerate(tau, start=1):
        # earliest timestep when dyn_sys is inside reachable set
        brs_at_t = V[..., t_earliest]

        # compute gradient
        index = grid.get_index(dyn_sys.x)
        gradient = spa_deriv(index, brs_at_t, grid, periodic_dims=[2])
        print(f"{gradient=}")

        print(gradient)
        # apply optimal control and disturbance
        for _ in range(subsamples):
            u = dyn_sys.opt_ctrl_non_hcl(_, dyn_sys.x, gradient)
            # d = dyn_sys.optDstb_non_hcl(_, dyn_sys.x, gradient)
            update_state(dyn_sys, u, 0, dt)

        if time_idx != V.shape[-1]:
            traj[time_idx] = dyn_sys.x

    return traj

