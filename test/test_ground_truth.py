import heterocl as hcl
import numpy as np
import pytest

from odp.common.derivatives import upwindFirstENO2, upwindFirstFirst
from odp.computeGraphs import graph_2D, graph_3D, graph_4D, graph_5D, graph_6D
from odp.Grid import Grid
from odp.solver import TTRSolver, solveValueIteration

# Compares the compiled HeteroCL kernels against dimension-agnostic NumPy ground truth on small grids.

GRAPHS = {2: graph_2D, 3: graph_3D, 4: graph_4D, 5: graph_5D, 6: graph_6D}


# =============================================================================
# Test system, usable by HeteroCL and NumPy:
#   dx_k/dt = speed(x) * u_k + d_k,  |u_k| <= 1,  -2 * dstb_bound(x) <= d_k <= dstb_bound(x)
# The control minimizes and the disturbance maximizes the Hamiltonian. The disturbance range is
# asymmetric so that mixing up its lower and upper bounds changes the dissipation.
# =============================================================================

def speed(x):
    return 1.0 + 0.1 * x[0] * x[0]


def dstb_bound(x, state_dependent):
    return 0.1 + 0.2 * x[-1] * x[-1] if state_dependent else 0.1


class System:
    def __init__(self, dims, state_dependent):
        self.dims = dims
        self.dstb_state_dependent = state_dependent

    def _bang(self, spat_deriv, if_positive, otherwise):
        out = []
        for k in range(self.dims):
            value = hcl.scalar(0, "value")
            with hcl.if_(spat_deriv[k] > 0):
                value[0] = if_positive
            with hcl.else_():
                value[0] = otherwise
            out.append(value[0])
        return tuple(out)

    def opt_ctrl(self, t, state, spat_deriv):
        return self._bang(spat_deriv, -1.0, 1.0)

    def opt_dstb(self, t, state, spat_deriv):
        bound = dstb_bound(state, self.dstb_state_dependent)
        return self._bang(spat_deriv, bound, -2.0 * bound)

    def dynamics(self, t, state, uOpt, dOpt):
        out = []
        for k in range(self.dims):
            value = hcl.scalar(0, "value")
            value[0] = speed(state) * uOpt[k] + dOpt[k]
            out.append(value[0])
        return tuple(out)


def np_ctrl(p):
    return [np.where(p_k > 0, -1.0, 1.0) for p_k in p]


def np_dstb(x, p, state_dependent):
    return [np.where(p_k > 0, 1.0, -2.0) * dstb_bound(x, state_dependent) for p_k in p]


def np_dynamics(x, u, d):
    return [speed(x) * u_k + d_k for u_k, d_k in zip(u, d)]


def make_grid(dims, pts, periodic=()):
    return Grid(np.array([-1.0] * dims), np.array([1.0] * dims), np.array([pts] * dims), list(periodic))


def states(grid):
    shape = tuple(grid.pts_each_dim)
    return [np.broadcast_to(v, shape) for v in grid.vs]


# =============================================================================
# Hamiltonian and dissipation
# =============================================================================

def ground_truth_rhs(grid, V, accuracy, state_dependent):
    """Returns the kernel output (Hamiltonian term + dissipation) and the CFL time step."""
    derivs = upwindFirstFirst if accuracy == "low" else upwindFirstENO2
    L, R = derivs(grid, V, generateAll=True)
    x = states(grid)

    p = [(l + r) / 2 for l, r in zip(L, R)]
    f = np_dynamics(x, np_ctrl(p), np_dstb(x, p, state_dependent))
    hamiltonian = sum(f_k * p_k for f_k, p_k in zip(f, p))

    # alpha_k: largest |f_k| over the control/disturbance at the global derivative bounds
    p_min = [min(l.min(), r.min()) for l, r in zip(L, R)]
    p_max = [max(l.max(), r.max()) for l, r in zip(L, R)]
    corners = [np_dynamics(x, np_ctrl(pu), np_dstb(x, pd, state_dependent))
               for pu in (p_min, p_max) for pd in (p_min, p_max)]
    alpha = [np.max([np.abs(c[k]) for c in corners], axis=0) for k in range(grid.dims)]

    dissipation = 0.5 * sum((r - l) * a for l, r, a in zip(L, R, alpha))
    dt = 0.8 / sum(a.max() / dx for a, dx in zip(alpha, grid.dx))
    return hamiltonian + dissipation, dt


@pytest.mark.parametrize("state_dependent", [True, False])
@pytest.mark.parametrize("accuracy", ["low", "medium"])
@pytest.mark.parametrize("dims, pts", [(2, 10), (3, 8), (4, 6), (5, 5), (6, 4)])
def test_hamiltonian_and_dissipation(dims, pts, accuracy, state_dependent):
    grid = make_grid(dims, pts, periodic=[dims - 1])
    V = np.random.default_rng(377).random(grid.pts_each_dim, dtype=np.float32)

    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)
    kernel = GRAPHS[dims](System(dims, state_dependent), grid, "minVWithV0", accuracy)
    rhs = hcl.asarray(np.zeros(V.shape, dtype=np.float32))
    delta_t = hcl.asarray(np.zeros(1, dtype=np.float32))
    xs = [hcl.asarray(np.reshape(v, n).astype(np.float32)) for v, n in zip(grid.vs, grid.pts_each_dim)]
    kernel(rhs, hcl.asarray(V), *xs, delta_t, hcl.asarray(np.zeros(1, dtype=np.float32)), hcl.asarray(V))

    expected_rhs, expected_dt = ground_truth_rhs(grid, V, accuracy, state_dependent)
    np.testing.assert_allclose(rhs.asnumpy(), expected_rhs, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(delta_t.asnumpy()[0], expected_dt, rtol=1e-5)


# =============================================================================
# Time-to-reach: the converged solution must satisfy the update equation
# =============================================================================

def ground_truth_ttr_update(grid, phi):
    L, R = upwindFirstFirst(grid, phi, generateAll=True)
    x = states(grid)
    p = [(l + r) / 2 for l, r in zip(L, R)]
    f = np_dynamics(x, np_ctrl(p), np_dstb(x, p, True))
    sigma = [np.abs(f_k) for f_k in f]

    hamiltonian = -(sum(f_k * p_k for f_k, p_k in zip(f, p)) + 1)
    c = sum(s / dx for s, dx in zip(sigma, grid.dx))
    dissipation = sum(s * ((r - l) / 2 + phi / dx) for s, l, r, dx in zip(sigma, L, R, grid.dx))
    return (-hamiltonian + dissipation) / c


@pytest.mark.parametrize("dims, pts", [(3, 9), (4, 7), (5, 5)])
def test_ttr_fixed_point(dims, pts):
    grid = make_grid(dims, pts)
    target = (np.sqrt(sum(v ** 2 for v in grid.vs)) - 0.6).astype(np.float32)
    phi = TTRSolver(System(dims, True), grid, target.copy(), 1e-5, verbose=False)

    interior = tuple(slice(1, -1) for _ in range(dims))
    phi_new = ground_truth_ttr_update(grid, phi)[interior]
    phi, target = phi[interior], target[interior]

    assert np.all(phi[target <= 0] == 0)
    assert target.max() > 0 and np.all(phi[target > 0] > 0) and np.all(phi < 1000)
    np.testing.assert_allclose(phi[target > 0], phi_new[target > 0], atol=1e-3)


# =============================================================================
# Value iteration
# As in odp/MDP_example, the actions are a flat 1D array and transition/reward receive one scalar u.
# Here u = +-(k + 1) moves one grid cell along axis k, in the direction given by the sign of u. The
# move succeeds with probability 0.8 and the state stays put otherwise. The next state is clamped to
# the grid.
# =============================================================================

GAMMA = 0.9
MOVE_PROBABILITY = 0.8


def flat_actions(dims):
    axes = np.arange(1.0, dims + 1)
    return np.concatenate([axes, -axes])


class GridMDP:
    def __init__(self, grid):
        self.dims = grid.dims
        self.dx = [float(dx) for dx in grid.dx]
        self.maxTransitions = 2

    def transition(self, sVals, u):
        trans_matrix = hcl.compute((self.maxTransitions, 1 + self.dims), lambda *x: 0, "trans_matrix")
        trans_matrix[0, 0] = MOVE_PROBABILITY
        trans_matrix[1, 0] = 1 - MOVE_PROBABILITY
        for k in range(self.dims):
            step = hcl.scalar(0, "step")
            with hcl.if_(u == k + 1):
                step[0] = self.dx[k]
            with hcl.elif_(u == -(k + 1)):
                step[0] = -self.dx[k]
            trans_matrix[0, k + 1] = sVals[k] + step[0]
            trans_matrix[1, k + 1] = sVals[k]
        return trans_matrix

    def reward(self, sVals, u):
        rwd = hcl.scalar(0, "rwd")
        rwd[0] = -0.01 * u
        for k in range(self.dims):
            rwd[0] = rwd[0] - sVals[k] * sVals[k]
        return rwd[0]


def ground_truth_value_iteration(grid, actions, iterations=400):
    shape = tuple(grid.pts_each_dim)
    idx = list(np.indices(shape))
    state_cost = sum(x_k ** 2 for x_k in states(grid))
    V = np.zeros(shape)
    for _ in range(iterations):
        Q = []
        for u in actions:
            axis = int(abs(u)) - 1
            moved = list(idx)
            moved[axis] = np.clip(idx[axis] + int(np.sign(u)), 0, shape[axis] - 1)
            Q.append(-0.01 * u - state_cost
                     + GAMMA * (MOVE_PROBABILITY * V[tuple(moved)] + (1 - MOVE_PROBABILITY) * V))
        V = np.max(Q, axis=0)
    return V


@pytest.mark.parametrize("dims, pts", [(3, 7), (4, 5), (5, 4)])
def test_value_iteration(dims, pts):
    grid = make_grid(dims, pts)
    actions = flat_actions(dims)
    V = solveValueIteration(GridMDP(grid), grid, actions, gamma=np.array([GAMMA]), epsilon=1e-5, maxIters=1000)
    np.testing.assert_allclose(V, ground_truth_value_iteration(grid, actions), atol=1e-3)
