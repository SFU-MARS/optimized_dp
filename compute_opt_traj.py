from Grid.GridProcessing import Grid
import numpy as np


# function [traj, traj_tau] = computeOptTraj(g, data, tau, dynSys, extraArgs)
# % Time parameters
# iter = 1;
# tauLength = length(tau);
# dtSmall = (tau(2) - tau(1))/subSamples;
# % maxIter = 1.25*tauLength;
#
# % Initialize trajectory
# traj = nan(g.dim, tauLength);
# traj(:,1) = dynSys.x;
# tEarliest = 1;
#
# while iter <= tauLength
#   % Determine the earliest time that the current state is in the reachable set
#   % Binary search
#   upper = tauLength;
#   lower = tEarliest;
#
#   % bug? tEarliest is always 1
#   tEarliest = find_earliest_BRS_ind(g, data, dynSys.x, upper, lower);
#
#   % BRS at current time
#   BRS_at_t = data(clns{:}, tEarliest);
#
#   % Visualize BRS corresponding to current trajectory point
#   if visualize
#     plot(traj(showDims(1), iter), traj(showDims(2), iter), 'k.')
#     hold on
#     [g2D, data2D] = proj(g, BRS_at_t, hideDims, traj(hideDims,iter));
#     visSetIm(g2D, data2D);
#     tStr = sprintf('t = %.3f; tEarliest = %.3f', tau(iter), tau(tEarliest));
#     title(tStr)
#     drawnow
#
#     if isfield(extraArgs, 'fig_filename')
#       export_fig(sprintf('%s%d', extraArgs.fig_filename, iter), '-png')
#     end
#
#     hold off
#   end
#
#   if tEarliest == tauLength
#     % Trajectory has entered the target
#     break
#   end
#
#   % Update trajectory
#   Deriv = computeGradients(g, BRS_at_t);
#   for j = 1:subSamples
#     deriv = eval_u(g, Deriv, dynSys.x);
#     u = dynSys.optCtrl(tau(tEarliest), dynSys.x, deriv, uMode);
#     d = dynSys.optDstb(tau(tEarliest), dynSys.x, deriv, dMode);
#     dynSys.updateState(u, dtSmall, dynSys.x, d);
#   end
#
#   % Record new point on nominal trajectory
#   iter = iter + 1;
#   traj(:,iter) = dynSys.x;
# end
#
# % Delete unused indices
# traj(:,iter:end) = [];
# traj_tau = tau(1:iter-1);
# end
def compute_gradients(g: Grid, time: int):
    pass


def find_earliest_brs_idx(g: Grid, V: np.ndarray, state: np.narray, low: int, high: int) -> int:
    """
    Determines the earliest time the current state is in the reachable set
    Args:
        g: Grid
        V: Value function
        state: state of dynamical system
        low: lower bound of search range (inclusive)
        high: upper bound of search range (inclusive)

    Returns:
        t: Earliest time where the state is in the reachable set
    """

    epsilon = 1e-4
    while low < high:
        mid = np.ceil((high + low) / 2)
        value = g.get_value(V[..., mid], state)
        if value < epsilon:
            low = mid
        else:
            high = mid - 1

    return low


def compute_opt_traj(g: Grid, V: np.ndarray, tau: np.ndarray, dyn_sys):
    """
    Computes the optimal trajectory for a dynamical system
    given

    Args:
        g: Grid that the value function was solved on
        V: Computed value function at each time step. The last dimension must be time.
        tau: Time [0, t]
        dynSys:

    Returns:

    """

    assert tuple(list(g.pts_each_dim)) == V.shape, (
        f"The length of tau does not match with the last dimension of V! "
        f"len(tau)={len(tau)}, V.shape={V.shape}"
    )

    opt_traj = np.nan(tau.shape)
    opt_u = np.nan(tau.shape)
    opt_d = np.nan(tau.shape)

    # earliest time dyn_sys is in the reachable set
    t_earliest = 0
    for i in range(len(tau)):
        lower = t_earliest
        upper = len(tau) - 1
        t_earliest = find_earliest_brs_idx(g, V, dyn_sys.x, lower, upper)

        # trajectory has entered the target
        if t_earliest == len(tau) - 1:
            break

        # deriv = compute_gradients(g, brs_at_t)

        for j in range(1):
            #     deriv = eval_u(g, Deriv, dynSys.x);
            u = dyn_sys.opt_ctrl(tau[t_earliest], dyn_sys.x)
            d = dynSys.optDstb(taurtEarliest], dyn_sys.x, deriv, dMode);
            dynSys.updateState(u, dtSmall, dynSys.x, d);

        # brs_at_time_t = V[..., t_earliest]?


if __name__ in "__main__":
    from user_definer import g, my_car, tau

    V = np.load("V_with_time.npy")
    compute_opt_traj(g, V, tau, my_car)
