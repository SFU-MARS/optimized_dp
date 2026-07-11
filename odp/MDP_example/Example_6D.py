"""
6-D Markov Decision Process example for OptimizedDP value iteration.

Run with::

    python -m odp.MDP_example.Example_6D

See :mod:`odp.MDP_example.Example_3D` for a description of the value-iteration
interface.  This is the reach-avoid template in six dimensions and is the
highest-dimensional MDP exposed through :func:`odp.solver.solveValueIteration`.
Note that value iteration is subject to the curse of dimensionality: keep the
per-dimension resolution and the action set modest at this dimension.
"""

import numpy as np
import heterocl as hcl

from odp.Grid import Grid
from odp.solver import solveValueIteration

_DIMS = 6


class MDP_6D_example:
    def __init__(self):
        self.bounds = np.array([[-2.5, 2.5]] * _DIMS)
        self.goal_center = np.zeros(_DIMS)
        self.goal_radius = 1.0
        self.wall = 0.2
        self.maxTransitions = 1

    def _dist_to_goal(self, sVals):
        acc = hcl.scalar(0, "acc")
        for d in range(_DIMS):
            diff = sVals[d] - self.goal_center[d]
            acc[0] += diff * diff
        return hcl.sqrt(acc[0])

    def _hits_wall(self, sVals):
        flag = hcl.scalar(0, "wall_flag")
        for d in range(_DIMS):
            with hcl.if_(hcl.or_(sVals[d] < self.bounds[d, 0] + self.wall,
                                 sVals[d] > self.bounds[d, 1] - self.wall)):
                flag[0] = 1
        return flag

    def transition(self, sVals, iVals, u):
        trans = hcl.compute((self.maxTransitions, 1 + _DIMS),
                            lambda *x: 0, "trans")
        mag = hcl.scalar(0, "mag")
        wall = self._hits_wall(sVals)
        mag[0] = self._dist_to_goal(sVals)
        with hcl.if_(hcl.or_(mag[0] <= self.goal_radius, wall[0] == 1)):
            trans[0, 0] = 0
        with hcl.else_():
            trans[0, 0] = 1.0
            for d in range(_DIMS):
                trans[0, 1 + d] = sVals[d] + u[d]
        return trans

    def reward(self, sVals, iVals, u):
        mag = hcl.scalar(0, "mag")
        rwd = hcl.scalar(0, "rwd")
        wall = self._hits_wall(sVals)
        with hcl.if_(wall[0] == 1):
            rwd[0] = -400
        with hcl.else_():
            mag[0] = self._dist_to_goal(sVals)
            with hcl.if_(mag[0] <= self.goal_radius):
                rwd[0] = 1000
            with hcl.else_():
                rwd[0] = 0
        return rwd[0]


def main():
    mdp = MDP_6D_example()
    grid = Grid(minBounds=mdp.bounds[:, 0], maxBounds=mdp.bounds[:, 1],
                dims=_DIMS, pts_each_dim=np.array([7] * _DIMS))
    step = np.linspace(-0.5, 0.5, 3)
    actions = np.array(np.meshgrid(*([step] * _DIMS))).T.reshape(-1, _DIMS)
    value_function = solveValueIteration(
        mdp, grid=grid, action_space=actions,
        gamma=np.array([0.93]), epsilon=np.array([0.3]),
        maxIters=np.array([50]))
    print("Value function shape:", value_function.shape)
    return value_function


if __name__ == "__main__":
    main()
