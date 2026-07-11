"""Unit tests for the implicit-surface shape helpers in :mod:`odp.Shapes`."""

import numpy as np

from odp.Grid import Grid
from odp.Shapes import CylinderShape, ShapeRectangle


def _grid_2d(n=41):
    return Grid(np.array([-4.0, -4.0]), np.array([4.0, 4.0]), 2, np.array([n, n]))


def test_cylinder_zero_level_set_matches_radius():
    """The implicit surface must vanish (change sign) at the given radius."""
    g = _grid_2d()
    radius = 2.0
    data = CylinderShape(g, ignore_dims=[], center=[0.0, 0.0], radius=radius,
                         quadratic=False)
    assert data.shape == tuple(g.pts_each_dim)
    # Centre is well inside the set (negative), corners are outside (positive).
    centre_idx = tuple(n // 2 for n in g.pts_each_dim)
    assert data[centre_idx] < 0
    assert data[0, 0] > 0
    # Linear signed-distance form: value == distance-to-centre minus radius.
    xx, yy = np.meshgrid(g.grid_points[0], g.grid_points[1], indexing="ij")
    expected = np.sqrt(xx ** 2 + yy ** 2) - radius
    np.testing.assert_allclose(data, expected, atol=1e-6)


def test_cylinder_ignore_dims_is_translation_invariant():
    """An ignored dimension must not influence the implicit surface."""
    g = Grid(np.array([-4.0, -4.0, -4.0]), np.array([4.0, 4.0, 4.0]),
             3, np.array([21, 21, 21]))
    data = CylinderShape(g, ignore_dims=[2], center=[0.0, 0.0, 0.0], radius=1.5)
    # Every slice along the ignored axis must be identical.
    for k in range(1, data.shape[2]):
        np.testing.assert_allclose(data[:, :, k], data[:, :, 0])


def test_rectangle_sign_inside_and_outside():
    g = _grid_2d()
    data = ShapeRectangle(g, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert data.shape == tuple(g.pts_each_dim)
    centre_idx = tuple(n // 2 for n in g.pts_each_dim)
    assert data[centre_idx] < 0     # inside the rectangle
    assert data[0, 0] > 0           # far corner is outside
