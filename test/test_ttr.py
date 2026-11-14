import heterocl as hcl
import numpy as np
import pytest

from odp.Grid import Grid
from odp.solver import TTRSolver

# Sanity checks for the 7D and 8D time-to-reach (TTR) solvers.
#
# Test system: single integrator dx_k/dt = u_k, |u_k| <= 1, where only the first ACTIVE dimensions
# move; the remaining dimensions have zero velocity. The box target ||(x_1, x_2)||_inf <= r only
# depends on the active dimensions, so the N-D time to reach must equal the 2D time to reach copied
# along the static dimensions, and is close to the analytic max(0, ||(x_1, x_2)||_inf - r).
# The static dimensions are periodic (3 points) except the last one, which is non-periodic and needs
# at least 4 points for the boundary extrapolation; this covers both boundary types.

ACTIVE = 2
ACTIVE_PTS = 21
STATIC_PTS = 3
LAST_DIM_PTS = 4
TARGET_RADIUS = 0.5
EPSILON = 1e-4


class Integrator:
    dstb_state_dependent = False

    def __init__(self, dims):
        self.dims = dims

    def opt_ctrl(self, t, state, spat_deriv):
        # Reach the target as fast as possible: u_k = -sign(dV/dx_k) in the active dimensions
        u = []
        for k in range(self.dims):
            u_k = hcl.scalar(0, "u_k")
            if k < ACTIVE:
                with hcl.if_(spat_deriv[k] > 0):
                    u_k[0] = -1.0
                with hcl.elif_(spat_deriv[k] < 0):
                    u_k[0] = 1.0
            u.append(u_k[0])
        return tuple(u)

    def opt_dstb(self, t, state, spat_deriv):
        d = []
        for k in range(self.dims):
            d_k = hcl.scalar(0, "d_k")
            d.append(d_k[0])
        return tuple(d)

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = []
        for k in range(self.dims):
            x_dot_k = hcl.scalar(0, "x_dot_k")
            x_dot_k[0] = uOpt[k]
            x_dot.append(x_dot_k[0])
        return tuple(x_dot)


def make_grid(dims):
    if dims == ACTIVE:
        return Grid(np.array([-1.0] * dims), np.array([1.0] * dims), np.array([ACTIVE_PTS] * dims))
    pts = [ACTIVE_PTS] * ACTIVE + [STATIC_PTS] * (dims - ACTIVE - 1) + [LAST_DIM_PTS]
    periodic = list(range(ACTIVE, dims - 1))
    return Grid(np.array([-1.0] * dims), np.array([1.0] * dims), np.array(pts), periodic)


def active_inf_norm(grid):
    shape = tuple(grid.pts_each_dim)
    return np.maximum(np.abs(np.broadcast_to(grid.vs[0], shape)), np.abs(np.broadcast_to(grid.vs[1], shape)))


def obstacle_fn(grid):
    # Obstacle: the strip x_1 >= 0.6 (negative inside the obstacle)
    return np.broadcast_to(0.6 - grid.vs[0], tuple(grid.pts_each_dim))


def solve(dims, with_obstacle=False):
    grid = make_grid(dims)
    target = (active_inf_norm(grid) - TARGET_RADIUS).astype(np.float32)
    values = target.copy()
    if with_obstacle:
        values = [values, obstacle_fn(grid).astype(np.float32)]
    ttr = TTRSolver(Integrator(dims), grid, values, EPSILON, verbose=False)
    return grid, target, ttr


@pytest.fixture(scope="module")
def reference_2d():
    return {obs: solve(ACTIVE, obs)[2] for obs in (False, True)}


@pytest.mark.parametrize("dims", [7, 8])
def test_ttr_matches_2d(dims, reference_2d):
    grid, target, ttr = solve(dims)
    dx = grid.dx[0]

    assert ttr.shape == tuple(grid.pts_each_dim)
    assert np.all(np.isfinite(ttr))
    # Points inside the target need no time; every other point reaches it in finite, positive time
    assert np.all(ttr[target <= 0] == 0)
    assert np.all(ttr[target > 0] > 0)
    assert np.all(ttr < 1000)

    # Constant along the static dimensions and equal to the 2D solution
    expected = reference_2d[False].reshape(ttr.shape[:ACTIVE] + (1,) * (dims - ACTIVE))
    np.testing.assert_allclose(ttr, np.broadcast_to(expected, ttr.shape), atol=1e-3)

    # Close to the analytic minimum time (first-order scheme on a 21-point grid)
    analytic = np.maximum(0.0, active_inf_norm(grid) - TARGET_RADIUS)
    np.testing.assert_allclose(ttr, analytic, atol=2 * dx)
    assert np.mean(np.abs(ttr - analytic)) < dx / 2

    # Symmetric under reversing either active dimension
    for d in range(ACTIVE):
        np.testing.assert_allclose(ttr, np.flip(ttr, axis=d), atol=1e-3,
                                   err_msg=f"TTR not symmetric along dimension {d}")


@pytest.mark.parametrize("dims", [7, 8])
def test_ttr_obstacle(dims, reference_2d):
    grid, target, ttr = solve(dims, with_obstacle=True)
    _, _, ttr_free = solve(dims)
    inside_obstacle = obstacle_fn(grid) <= 0

    assert inside_obstacle.any()
    # Obstacle cells stay unreachable
    assert np.all(ttr[inside_obstacle] == 1000)
    # Obstacles can only increase the time to reach everywhere else
    assert np.all(ttr[~inside_obstacle] >= ttr_free[~inside_obstacle] - 1e-5)
    # Equal to the 2D solution with the same obstacle
    expected = reference_2d[True].reshape(ttr.shape[:ACTIVE] + (1,) * (dims - ACTIVE))
    np.testing.assert_allclose(ttr, np.broadcast_to(expected, ttr.shape), atol=1e-3)
