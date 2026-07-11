"""Regression and sanity tests for MDP value iteration (`solveValueIteration`)."""

import os

import numpy as np

from reference_problems import solve_mdp_2d, solve_mdp_3d

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_mdp_matches_reference():
    result = solve_mdp_2d()
    reference = np.load(os.path.join(DATA_DIR, "mdp_2d.npy"))
    assert result.shape == (31, 41)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, reference, rtol=1e-3, atol=1e-2)


def test_mdp_value_function_properties():
    """With a strictly non-positive reward, the optimal value function is
    non-positive everywhere and is maximised (== 0) at the upright, zero-
    velocity state (the grid centre)."""
    result = solve_mdp_2d()
    assert result.max() <= 1e-6
    centre = (result.shape[0] // 2, result.shape[1] // 2)
    # The upright state should have the largest (closest-to-zero) value.
    assert result[centre] == np.max(result)


def test_mdp_multidimensional_action_space():
    """Value iteration with a 2-component (v, w) action space must run and
    propagate reward via the discount factor (regression test for the
    ``intermeds`` allocation bug that broke multi-dimensional action spaces)."""
    result = solve_mdp_3d()
    reference = np.load(os.path.join(DATA_DIR, "mdp_3d.npy"))
    assert result.shape == (25, 25, 9)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-3)
    # Reward propagates: the value function takes more than the two trivial
    # {obstacle penalty, goal reward} levels once discounting kicks in.
    assert np.unique(np.round(result, 1)).size > 3
