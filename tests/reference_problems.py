"""
Shared, small-scale problem definitions used by the OptimizedDP test suite.

Each ``solve_*`` function builds a *small* grid problem (so it runs in a few
seconds) and returns the resulting value function as a NumPy array.  The very
same functions are used both to

  * regenerate the golden reference arrays in ``tests/data`` (see
    ``regenerate_references.py``), and
  * produce the arrays that the regression tests compare against those
    references.

Keeping the two in one place guarantees that the reference data and the tested
computation never drift apart.
"""

import math

import heterocl as hcl
import numpy as np

from odp.Grid import Grid
from odp.Shapes import CylinderShape, ShapeRectangle
from odp.dynamics import Plane1D, Plane2D, DubinsCapture, DubinsCar
from odp.solver import HJSolver, TTRSolver, solveValueIteration


# --------------------------------------------------------------------------- #
#  Hamilton-Jacobi reachability problems
# --------------------------------------------------------------------------- #
def solve_hj_1d():
    """1D backward reachable tube for the (trivial) Plane1D system."""
    g = Grid(np.array([-4.0]), np.array([4.0]), 1, np.array([40]))
    init = ShapeRectangle(g, np.array([-1.0]), np.array([1.0]))
    tau = np.arange(start=0, stop=0.5 + 1e-5, step=0.05)
    return HJSolver(Plane1D(), g, init, tau, {"TargetSetMode": "None"},
                    saveAllTimeSteps=False)


def solve_hj_2d():
    """2D backward reachable tube for the Plane2D system."""
    g = Grid(np.array([-4.0, -4.0]), np.array([4.0, 4.0]), 2, np.array([40, 40]))
    init = ShapeRectangle(g, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    tau = np.arange(start=0, stop=0.5 + 1e-5, step=0.05)
    return HJSolver(Plane2D(), g, init, tau, {"TargetSetMode": "None"},
                    saveAllTimeSteps=False)


def solve_hj_3d():
    """3D differential game (DubinsCapture) backward reachable tube."""
    g = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]),
             3, np.array([30, 30, 20]), [2])
    init = CylinderShape(g, [2], np.zeros(3), 1.0)
    tau = np.arange(start=0, stop=0.5 + 1e-5, step=0.05)
    return HJSolver(DubinsCapture(uMode="max", dMode="min"), g, init, tau,
                    {"TargetSetMode": "None"}, saveAllTimeSteps=False)


# --------------------------------------------------------------------------- #
#  Time-to-reach problem
# --------------------------------------------------------------------------- #
def solve_ttr_3d():
    """3D time-to-reach for a Dubins car reaching a cylindrical target."""
    g = Grid(np.array([-3.0, -3.0, -math.pi]), np.array([3.0, 3.0, math.pi]),
             3, np.array([25, 25, 15]), [2])
    target = CylinderShape(g, [2], np.array([0.0, 0.0, 0.0]), 0.7)
    return TTRSolver(DubinsCar(uMode="min"), g, target, epsilon=0.01)


# --------------------------------------------------------------------------- #
#  Markov decision process (value iteration)
# --------------------------------------------------------------------------- #
class _Pendulum2D:
    """Minimal deterministic inverted-pendulum MDP (OpenAI-gym dynamics)."""

    def __init__(self):
        self.dt = 0.05
        self.gravity = 10.0
        self.m = 1.0
        self.l = 1.0
        self.max_speed = 8.0
        self.coeff1 = 3 * self.gravity / (2 * self.l)
        self.coeff2 = 3.0 / (self.m * self.l * self.l)
        self.maxTransitions = 1

    def transition(self, sVals, iVals, u):
        trans = hcl.compute((self.maxTransitions, 3), lambda *x: 0, "trans")
        newthdot = hcl.scalar(0, "newthdot")
        new_th = hcl.scalar(0, "new_th")
        newthdot[0] = sVals[1] + (self.coeff1 * hcl.sin(sVals[0])
                                  + self.coeff2 * u) * self.dt
        with hcl.if_(newthdot[0] > self.max_speed):
            newthdot[0] = self.max_speed
        with hcl.if_(newthdot[0] < -self.max_speed):
            newthdot[0] = -self.max_speed
        new_th[0] = sVals[0] + newthdot[0] * self.dt
        with hcl.if_(new_th[0] >= math.pi):
            new_th[0] = new_th[0] - 2 * math.pi
        with hcl.elif_(new_th[0] < -math.pi):
            new_th[0] = new_th[0] + 2 * math.pi
        trans[0, 0] = 1.0
        trans[0, 1] = new_th[0]
        trans[0, 2] = newthdot[0]
        return trans

    def reward(self, sVals, iVals, u):
        rwd = hcl.scalar(0, "rwd")
        rwd[0] = -(sVals[0] * sVals[0] + 0.1 * sVals[1] * sVals[1]
                   + 0.001 * u * u)
        return rwd[0]


def solve_mdp_2d():
    """Value iteration on a small inverted-pendulum grid (1-D action space)."""
    g = Grid(minBounds=np.array([-math.pi, -8.0]),
             maxBounds=np.array([math.pi, 8.0]),
             dims=2, pts_each_dim=np.array([31, 41]))
    return solveValueIteration(_Pendulum2D(), grid=g,
                               action_space=np.linspace(-2.0, 2.0, 21),
                               gamma=np.array([0.99]),
                               epsilon=np.array([1e-3]),
                               maxIters=np.array([200]))


def solve_mdp_3d():
    """Value iteration on the 3-D reach-avoid example.

    Exercises a *multi-dimensional* (2-component) action space, i.e. the code
    path where each action is a ``(v, w)`` pair rather than a scalar.
    """
    from odp.MDP_example.Example_3D import MDP_3D_example
    mdp = MDP_3D_example()
    g = Grid(minBounds=mdp.bounds[:, 0], maxBounds=mdp.bounds[:, 1],
             dims=3, pts_each_dim=np.array([25, 25, 9]), periodicDims=[2])
    actions = np.array([(v, w)
                        for v in np.linspace(-2.0, 2.0, 9)
                        for w in np.linspace(-1.0, 1.0, 9)])
    return solveValueIteration(mdp, grid=g, action_space=actions,
                               gamma=np.array([0.93]),
                               epsilon=np.array([0.3]),
                               maxIters=np.array([500]))


# Registry consumed by both the regeneration script and the tests.
PROBLEMS = {
    "hj_1d": solve_hj_1d,
    "hj_2d": solve_hj_2d,
    "hj_3d": solve_hj_3d,
    "ttr_3d": solve_ttr_3d,
    "mdp_2d": solve_mdp_2d,
    "mdp_3d": solve_mdp_3d,
}
