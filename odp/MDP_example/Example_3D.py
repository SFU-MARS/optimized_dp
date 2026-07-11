"""
3-D Markov Decision Process example for OptimizedDP value iteration.

This file documents, and provides a runnable example of, the **current**
MDP / value-iteration interface used by :func:`odp.solver.solveValueIteration`.

Run it directly with::

    python -m odp.MDP_example.Example_3D

--------------------------------------------------------------------------------
VALUE-ITERATION INTERFACE
--------------------------------------------------------------------------------
The user supplies a plain Python object exposing two methods and one attribute:

    maxTransitions : int
        The maximum number of successor states that ``transition`` can return
        for any (state, action) pair.  For a deterministic system this is 1.

    transition(self, sVals, iVals, u) -> hcl.compute tensor
        Returns a matrix of shape ``(maxTransitions, 1 + n_dims)`` describing the
        possible successor states of taking action ``u`` from state ``sVals``:
            row i, column 0            : probability of transition i
            row i, columns 1 .. n_dims : the successor state values of transition i
        The probabilities in column 0 should sum to 1 (or to 0 to indicate an
        absorbing / terminal state that receives no future reward).

    reward(self, sVals, iVals, u) -> hcl.scalar value
        The immediate reward for taking action ``u`` from state ``sVals``.

    Method arguments:
        sVals : the *continuous* state values, sVals[k] = value of dimension k.
        iVals : the *integer grid indices* of that state (iVals[k] = index of
                dimension k).  Provided for convenience; unused here.
        u     : the action, one element of ``action_space``.

The state space, grid resolution, discount factor, convergence tolerance and
iteration cap are passed to :func:`solveValueIteration` (see ``__main__`` below);
they are no longer declared as class attributes.
"""

import math

import numpy as np
import heterocl as hcl

from odp.Grid import Grid
from odp.solver import solveValueIteration


class MDP_3D_example:
    """A Dubins-car-style reach-avoid MDP on a 3-D (x, y, theta) grid."""

    def __init__(self):
        # Absorbing goal region: position within radius 1 of (3.5, 3.5) with a
        # heading between pi/2 and 3pi/4.
        self.goal_xy = np.array([3.5, 3.5])
        self.goal_radius = 1.0
        self.goal_theta = np.array([1.5707, 2.3562])
        # State-space bounds (must match the Grid created in __main__).
        self.bounds = np.array([[-5.0, 5.0],
                                [-5.0, 5.0],
                                [-math.pi, math.pi]])
        self.wall = 0.2          # obstacle margin next to the domain walls
        self.step = 0.6          # integration step for the deterministic move
        self.maxTransitions = 1  # deterministic dynamics -> single successor

    # --- required interface -------------------------------------------------
    def transition(self, sVals, iVals, u):
        trans = hcl.compute((self.maxTransitions, 1 + 3), lambda *x: 0, "trans")

        dx = hcl.scalar(0, "dx")
        dy = hcl.scalar(0, "dy")
        mag = hcl.scalar(0, "mag")
        dx[0] = sVals[0] - self.goal_xy[0]
        dy[0] = sVals[1] - self.goal_xy[1]
        mag[0] = hcl.sqrt(dx[0] * dx[0] + dy[0] * dy[0])

        # Terminal (absorbing) states get probability 0 -> no future reward.
        with hcl.if_(hcl.and_(mag[0] <= self.goal_radius,
                              sVals[2] <= self.goal_theta[1],
                              sVals[2] >= self.goal_theta[0])):
            trans[0, 0] = 0
        with hcl.elif_(hcl.or_(sVals[0] < self.bounds[0, 0] + self.wall,
                               sVals[0] > self.bounds[0, 1] - self.wall)):
            trans[0, 0] = 0
        with hcl.elif_(hcl.or_(sVals[1] < self.bounds[1, 0] + self.wall,
                               sVals[1] > self.bounds[1, 1] - self.wall)):
            trans[0, 0] = 0
        # Standard deterministic move.
        with hcl.else_():
            trans[0, 0] = 1.0
            trans[0, 1] = sVals[0] + self.step * u[0] * hcl.cos(sVals[2])
            trans[0, 2] = sVals[1] + self.step * u[0] * hcl.sin(sVals[2])
            trans[0, 3] = sVals[2] + self.step * u[1]
            # Wrap the periodic heading dimension back into [-pi, pi).
            with hcl.while_(trans[0, 3] > math.pi):
                trans[0, 3] -= 2 * math.pi
            with hcl.while_(trans[0, 3] < -math.pi):
                trans[0, 3] += 2 * math.pi
        return trans

    def reward(self, sVals, iVals, u):
        dx = hcl.scalar(0, "dx")
        dy = hcl.scalar(0, "dy")
        mag = hcl.scalar(0, "mag")
        rwd = hcl.scalar(0, "rwd")

        with hcl.if_(hcl.or_(sVals[0] < self.bounds[0, 0] + self.wall,
                             sVals[0] > self.bounds[0, 1] - self.wall)):
            rwd[0] = -400
        with hcl.elif_(hcl.or_(sVals[1] < self.bounds[1, 0] + self.wall,
                               sVals[1] > self.bounds[1, 1] - self.wall)):
            rwd[0] = -400
        with hcl.else_():
            dx[0] = sVals[0] - self.goal_xy[0]
            dy[0] = sVals[1] - self.goal_xy[1]
            mag[0] = hcl.sqrt(dx[0] * dx[0] + dy[0] * dy[0])
            with hcl.if_(hcl.and_(mag[0] <= self.goal_radius,
                                  sVals[2] <= self.goal_theta[1],
                                  sVals[2] >= self.goal_theta[0])):
                rwd[0] = 1000
            with hcl.else_():
                rwd[0] = 0
        return rwd[0]


def main():
    mdp = MDP_3D_example()
    grid = Grid(minBounds=mdp.bounds[:, 0],
                maxBounds=mdp.bounds[:, 1],
                dims=3,
                pts_each_dim=np.array([25, 25, 9]),
                periodicDims=[2])

    # Action space: (linear velocity, angular velocity) pairs.
    v_values = np.linspace(-2.0, 2.0, 9)
    w_values = np.linspace(-1.0, 1.0, 9)
    actions = np.array([(v, w) for v in v_values for w in w_values])

    value_function = solveValueIteration(
        mdp,
        grid=grid,
        action_space=actions,
        gamma=np.array([0.93]),
        epsilon=np.array([0.3]),
        maxIters=np.array([500]),
    )
    print("Value function shape:", value_function.shape)
    return value_function


if __name__ == "__main__":
    main()
