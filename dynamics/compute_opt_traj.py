from Grid.GridProcessing import Grid
from spatialDerivatives.first_order_generic import spa_deriv
import numpy as np


def compute_opt_traj(grid: Grid, V, tau, dyn_sys):
    assert V.shape[-1] == len(tau)

    dt = tau[1] - tau[0]

    # traj stores state of dyn_sys from look_back_length -> 0
    # first entry is dyn_sys at time -t
    # second entry is dyn_sys at time -t + delta t
    # traj(t, x)
    traj = np.empty((V.shape[-1], len(dyn_sys.x)))
    traj[0] = dyn_sys.x
    t_earliest = 0

    for time_idx, time in enumerate(tau):
        # earliest timestep when dyn_sys is inside reachable set
        brs_at_t = V[..., t_earliest]

        # compute gradient
        index = grid.get_index(dyn_sys.x)
        gradient = spa_deriv(index, V, grid)

        # apply optimal control and disturbance
        for _ in range(5):
            u = dyn_sys.opt_ctrl(_, dyn_sy.x, gradient)
            d = dyn_sys.optDstb(_, dyn_sy.x, gradient)
