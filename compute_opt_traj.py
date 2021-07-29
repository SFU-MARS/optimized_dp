from Grid.GridProcessing import Grid
import numpy as np
from scipy.integrate import solve_ivp
from dynamics.DubinsCapture import DubinsCapture


def compute_gradients(g: Grid, time: int):
    pass


def find_earliest_brs_idx(g: Grid, V: np.ndarray, state: np.ndarray, low: int, high: int) -> int:
    """
    Determines the earliest time the current state is in the reachable set
    Args:
        g: Grid
        V: Value function
        state: state of dynamical system
        low: lower bound of search range (inclusive)
        high: upper bound of search range (inclusive)

    Returns:
        t: Earliest time where the state is in the reachable set (when V(x) <= 0)
    """

    epsilon = 1e-4
    while low < high:
        mid = int(np.ceil((high + low) / 2))
        value = g.get_value(V[..., mid], state)
        if value < epsilon:
            low = mid
        else:
            high = mid - 1

    return high


def compute_opt_traj_3d(grid: Grid, V: np.ndarray, tau: np.ndarray, dyn_sys):
    """
    Computes the optimal trajectory for a given dynamical system
    Not build hcl functions

    Args:
        grid: Grid that the value function was solved on
        V: Computed value function at each time step. The last dimension must be time.
        tau: linspace from [0, T]
        dyn_sys: Dynamic System Object

    Returns:

    """

    assert tuple(list(g.pts_each_dim) + [len(tau)]) == V.shape, (
        f"The length of tau does not match with the last dimension of V! "
        f"len(tau)={len(tau)}, grid={g.pts_each_dim}"
        f"V.shape={V.shape}"
    )

    value = grid.get_value(V[..., -1], dyn_sys.x)

    if value > 0.0:
        print(f"Initial state does not in the BRT/BRS. It has the value of {value:.2f}")
        return None


    # TODO: remove hardcoded sizes for traj, u, d
    opt_traj = np.full([len(tau), 3], np.nan)
    opt_u = np.full([len(tau), 1], np.nan)
    opt_d = np.full([len(tau), 1], np.nan)

    # earliest time dyn_sys is in the reachable set
    t_earliest = 0
    for i in range(1):
        lower = t_earliest
        upper = len(tau) - 1
        t_earliest = find_earliest_brs_idx(g, V, dyn_sys.x, lower, upper)

        # trajectory has entered the target
        if t_earliest == len(tau) - 1:
            break
        #
        # # deriv = compute_gradients(g, brs_at_t)
        #
        # for _ in range(4):
        #     # deriv = eval_u(g, Deriv, dynSys.x);
        #     # u = dyn_sys.opt_ctrl(tau[t_earliest], dyn_sys.x)
        #     # d = dynSys.optDstb(taurtEarliest], dyn_sys.x, deriv, dMode);
        #     dynSys.updateState(u, dtSmall, dynSys.x, d);
        #
        # # brs_at_time_t = V[..., t_earliest]?


def update_dynamics(dyn_sys, u, d):
    sol = solve_ivp(dyn_sys.dynamics_non_hcl, [0, 0.1], dyn_sys.x, args=(u, d))
    dyn_sys.x = sol.y[:, -1]


if __name__ in "__main__":
    from user_definer import g, tau
    dyn_sys = DubinsCapture(x=np.array([2, 2, -np.pi]))
    V = np.load("V_with_time.npy")
    compute_opt_traj_3d(g, V, tau, dyn_sys)
    # update_dynamics(dyn_sys, 1, 0)
