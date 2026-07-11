"""Unit tests for the :class:`odp.Grid.Grid` helper."""

import math

import numpy as np
import pytest

from odp.Grid import Grid


def test_grid_basic_attributes():
    g = Grid(np.array([-4.0, -4.0]), np.array([4.0, 4.0]), 2, np.array([41, 21]))
    assert g.dims == 2
    assert tuple(g.pts_each_dim) == (41, 21)
    # dx = (max - min) / (N - 1)
    np.testing.assert_allclose(g.dx, [8.0 / 40, 8.0 / 20])
    assert len(g.vs) == 2
    assert len(g.grid_points) == 2
    assert g.grid_points[0][0] == pytest.approx(-4.0)
    assert g.grid_points[0][-1] == pytest.approx(4.0)


def test_grid_accepts_plain_python_lists():
    """The constructor must accept lists, not only numpy arrays.

    (Directly answers a reviewer question about the README example.)
    """
    g_list = Grid([-4.0, -4.0], [4.0, 4.0], 2, [41, 21])
    g_arr = Grid(np.array([-4.0, -4.0]), np.array([4.0, 4.0]), 2,
                 np.array([41, 21]))
    np.testing.assert_array_equal(g_list.min, g_arr.min)
    np.testing.assert_array_equal(g_list.max, g_arr.max)
    np.testing.assert_array_equal(g_list.pts_each_dim, g_arr.pts_each_dim)
    np.testing.assert_allclose(g_list.dx, g_arr.dx)


def test_grid_infers_dims_when_omitted():
    """`dims` is redundant with the bound vectors and may be omitted."""
    g = Grid(np.array([-4.0, -4.0, -2.0]), np.array([4.0, 4.0, 2.0]),
             pts_each_dim=np.array([21, 21, 11]))
    assert g.dims == 3
    assert tuple(g.pts_each_dim) == (21, 21, 11)


def test_grid_dimension_mismatch_raises():
    # dims disagrees with the length of the bound / point vectors.
    with pytest.raises(AssertionError):
        Grid(np.array([-4.0, -4.0]), np.array([4.0, 4.0]), 3, np.array([41, 21]))


def test_periodic_dimension_excludes_upper_bound():
    """For a periodic dimension the grid spans [min, max) (upper end dropped)."""
    n = 20
    g = Grid(np.array([-4.0, -math.pi]), np.array([4.0, math.pi]),
             2, np.array([30, n]), periodicDims=[1])
    span = 2 * math.pi
    expected_max = -math.pi + span * (1 - 1.0 / n)
    assert g.max[1] == pytest.approx(expected_max)
    # The non-periodic dimension keeps its upper bound.
    assert g.max[0] == pytest.approx(4.0)
