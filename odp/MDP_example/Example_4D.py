"""
4-D Markov Decision Process example for OptimizedDP value iteration.

Run with::

    python -m odp.MDP_example.Example_4D

See :mod:`odp.MDP_example.Example_3D` for a full description of the
value-iteration interface (``maxTransitions``, ``transition`` and ``reward``).
This example uses the same reach-avoid template in four dimensions: the state
is a point in R^4, each action is a per-dimension displacement, moving into the
domain walls is penalised, and a ball around the origin is an absorbing goal.
"""

import numpy as np
import heterocl as hcl

from odp.Grid import Grid
from odp.solver import solveValueIteration


class MDP_4D_example:
    def __init__(self):
        self.bounds = np.array([[-2.5, 2.5]] * 4)
        self.goal_center = np.zeros(4)
        self.goal_radius = 1.0
        self.wall = 0.2
        self.maxTransitions = 1

    def _dist_to_goal(self, sVals):
        acc = hcl.scalar(0, "acc")
        for d in range(4):
            diff = sVals[d] - self.goal_center[d]
            acc[0] += diff * diff
        return hcl.sqrt(acc[0])

    def transition(self, sVals, iVals, u):
        trans = hcl.compute((self.maxTransitions, 1 + 4), lambda *x: 0, "trans")
        mag = hcl.scalar(0, "mag")
        mag[0] = self._dist_to_goal(sVals)
        with hcl.if_(mag[0] <= self.goal_radius):
            trans[0, 0] = 0
        with hcl.elif_(hcl.or_(sVals[0] < self.bounds[0, 0] + self.wall,
                               sVals[0] > self.bounds[0, 1] - self.wall,
                               sVals[1] < self.bounds[1, 0] + self.wall,
                               sVals[1] > self.bounds[1, 1] - self.wall)):
            trans[0, 0] = 0
        with hcl.elif_(hcl.or_(sVals[2] < self.bounds[2, 0] + self.wall,
                               sVals[2] > self.bounds[2, 1] - self.wall,
                               sVals[3] < self.bounds[3, 0] + self.wall,
                               sVals[3] > self.bounds[3, 1] - self.wall)):
            trans[0, 0] = 0
        with hcl.else_():
            trans[0, 0] = 1.0
            for d in range(4):
                trans[0, 1 + d] = sVals[d] + u[d]
        return trans

    def reward(self, sVals, iVals, u):
        mag = hcl.scalar(0, "mag")
        rwd = hcl.scalar(0, "rwd")
        with hcl.if_(hcl.or_(sVals[0] < self.bounds[0, 0] + self.wall,
                             sVals[0] > self.bounds[0, 1] - self.wall,
                             sVals[1] < self.bounds[1, 0] + self.wall,
                             sVals[1] > self.bounds[1, 1] - self.wall,
                             sVals[2] < self.bounds[2, 0] + self.wall,
                             sVals[2] > self.bounds[2, 1] - self.wall,
                             sVals[3] < self.bounds[3, 0] + self.wall,
                             sVals[3] > self.bounds[3, 1] - self.wall)):
            rwd[0] = -400
        with hcl.else_():
            mag[0] = self._dist_to_goal(sVals)
            with hcl.if_(mag[0] <= self.goal_radius):
                rwd[0] = 1000
            with hcl.else_():
                rwd[0] = 0
        return rwd[0]


def main():
    mdp = MDP_4D_example()
    grid = Grid(minBounds=mdp.bounds[:, 0], maxBounds=mdp.bounds[:, 1],
                dims=4, pts_each_dim=np.array([15, 15, 15, 15]))
    step = np.linspace(-0.5, 0.5, 3)
    actions = np.array([(a, b, c, d)
                        for a in step for b in step
                        for c in step for d in step])
    value_function = solveValueIteration(
        mdp, grid=grid, action_space=actions,
        gamma=np.array([0.93]), epsilon=np.array([0.3]),
        maxIters=np.array([50]))
    print("Value function shape:", value_function.shape)
    return value_function


if __name__ == "__main__":
    main()
