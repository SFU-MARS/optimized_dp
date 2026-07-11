"""Regression and sanity tests for the Hamilton-Jacobi solver (`HJSolver`).

The regression tests compare freshly computed value functions against golden
reference arrays stored in ``tests/data`` (regenerate with
``python tests/regenerate_references.py`` after an intentional algorithm
change).
"""

import os

import numpy as np
import pytest

from reference_problems import solve_hj_1d, solve_hj_2d, solve_hj_3d

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load(name):
    return np.load(os.path.join(DATA_DIR, f"{name}.npy"))


@pytest.mark.parametrize("name, solve, expected_shape", [
    ("hj_1d", solve_hj_1d, (40,)),
    ("hj_2d", solve_hj_2d, (40, 40)),
    ("hj_3d", solve_hj_3d, (30, 30, 20)),
])
def test_hj_matches_reference(name, solve, expected_shape):
    result = solve()
    reference = _load(name)
    assert result.shape == expected_shape
    assert np.all(np.isfinite(result))
    # Tolerances chosen to absorb float32 / parallel-reduction noise while
    # still catching any real change in the numerical result.
    np.testing.assert_allclose(result, reference, rtol=1e-3, atol=1e-3)


def test_hj_backward_reachable_tube_is_monotone():
    """A backward reachable *tube* can only grow: the value at each grid point
    must not increase relative to the initial (target) function."""
    from odp.Grid import Grid
    from odp.Shapes import ShapeRectangle
    from odp.dynamics import Plane2D
    from odp.solver import HJSolver
    import numpy as np

    g = Grid(np.array([-4.0, -4.0]), np.array([4.0, 4.0]), 2, np.array([40, 40]))
    init = ShapeRectangle(g, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    tau = np.arange(start=0, stop=0.5 + 1e-5, step=0.05)
    result = HJSolver(Plane2D(), g, init, tau, {"TargetSetMode": "None"},
                      saveAllTimeSteps=False)
    # Reachable tube (min with target over time) never exceeds the target.
    assert np.all(result <= init + 1e-6)
