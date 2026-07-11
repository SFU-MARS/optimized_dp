"""Regression and sanity tests for the time-to-reach solver (`TTRSolver`)."""

import os

import numpy as np

from reference_problems import solve_ttr_3d

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_ttr_matches_reference():
    result = solve_ttr_3d()
    reference = np.load(os.path.join(DATA_DIR, "ttr_3d.npy"))
    assert result.shape == (25, 25, 15)
    assert np.all(np.isfinite(result))
    # float32 + parallel reductions give run-to-run noise at the ~1e-3 level;
    # a genuine algorithmic regression would shift values far more than atol.
    np.testing.assert_allclose(result, reference, rtol=1e-3, atol=2e-2)


def test_ttr_values_are_nonnegative_and_zero_at_target():
    """Time-to-reach is non-negative everywhere and (near) zero inside the
    target region located at the grid centre."""
    result = solve_ttr_3d()
    assert result.min() >= 0.0
    # The cylindrical target is centred at the origin -> centre index.
    centre = (result.shape[0] // 2, result.shape[1] // 2, result.shape[2] // 2)
    assert result[centre] < 0.5
