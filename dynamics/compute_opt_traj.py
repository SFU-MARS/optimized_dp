from Grid.GridProcessing import Grid


def compute_opt_traj(grid: Grid, V, tau, dyn_sys):
    dt = tau[1] - tau[0]
    t_earliest = 0
    for _ in tau:
        # earliest timestep when dyn_sys is inside reachable set
        brs_at_t = V[..., t_earliest]

        # compute gradient
        index = grid.get_index(dyn_sys.x)

        # apply optimal control and disturbance


