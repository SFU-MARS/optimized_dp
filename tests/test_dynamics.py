"""Unit tests for the dynamical-system models in :mod:`odp.dynamics`.

The solver tests exercise a handful of systems *indirectly* (as inputs to an HJ
or TTR solve).  This module tests the dynamics classes *directly* in two ways:

1. **Instantiation / export smoke tests** -- every class advertised by
   ``odp.dynamics`` can be imported and constructed with its defaults.  This is
   the direct regression guard for Referee 1's comment m2 (``__init__.py`` used
   to import a non-existent ``DubinsCar5D``; the real 5-D system is
   ``DubinsCar5DAvoid``).

2. **Pure-Python method assertions** -- the ``dynamics_inPython`` /
   ``optCtrl_inPython`` / ``forward`` helpers are plain NumPy (they do *not* run
   inside a HeteroCL kernel), so their numerical behaviour can be checked
   exactly and cheaply.  These helpers are also the most recently changed part
   of the dynamics package and previously had no coverage at all.
"""

import math

import numpy as np
import pytest

import odp.dynamics as dynamics
from odp.dynamics import (
    DubinsAirplane6D,
    DubinsCapture,
    DubinsCar,
    DubinsCar2,
    DubinsCar4D,
    DubinsCar4D2,
    DubinsCar5DAvoid,
    Plane1D,
    Plane2D,
)


# --------------------------------------------------------------------------- #
#  1. Instantiation / export smoke tests
# --------------------------------------------------------------------------- #
ALL_DYNAMICS = [
    DubinsCapture,
    DubinsCar,
    DubinsCar2,
    DubinsCar4D,
    DubinsCar4D2,
    DubinsCar5DAvoid,
    DubinsAirplane6D,
    Plane1D,
    Plane2D,
]


@pytest.mark.parametrize("cls", ALL_DYNAMICS, ids=lambda c: c.__name__)
def test_dynamics_class_instantiates_with_defaults(cls):
    """Every exported dynamics class must be constructible with no arguments
    and expose the common ``x`` / ``uMode`` interface."""
    system = cls()
    assert hasattr(system, "x")
    assert hasattr(system, "uMode")
    assert system.uMode in ("min", "max")


def test_dynamics_package_exports_only_existing_classes():
    """Regression guard for Referee 1's comment m2.

    ``odp.dynamics`` must export exactly the systems that exist -- in
    particular the 5-D system is ``DubinsCar5DAvoid``; the non-existent
    ``DubinsCar5D`` that ``__init__.py`` used to reference must not reappear.
    """
    for cls in ALL_DYNAMICS:
        assert hasattr(dynamics, cls.__name__)
    assert not hasattr(dynamics, "DubinsCar5D")


# --------------------------------------------------------------------------- #
#  2. Pure-Python method assertions
# --------------------------------------------------------------------------- #
#  Dubins car (3-D, single turn-rate control)
# --------------------------------------------------------------------------- #
def test_dubinscar_dynamics_inPython():
    car = DubinsCar()  # speed=1, uMode="min"
    # Heading 0 -> velocity is purely along +x.
    dx, dy, dtheta = car.dynamics_inPython(state=[0.0, 0.0, 0.0], action=[0.5])
    assert (dx, dy, dtheta) == pytest.approx((1.0, 0.0, 0.5))
    # Heading pi/2 -> velocity is purely along +y.
    dx, dy, _ = car.dynamics_inPython(state=[0.0, 0.0, math.pi / 2], action=[0.0])
    assert dx == pytest.approx(0.0, abs=1e-12)
    assert dy == pytest.approx(1.0)


def test_dubinscar_forward_euler_step_and_angle_wrap():
    car = DubinsCar()
    nx, ny, nth = car.forward(ctrl_freq=10, current_state=[0.0, 0.0, 0.0], u=0.0)
    assert (nx, ny, nth) == pytest.approx((0.1, 0.0, 0.0))
    # A negative resulting heading must be wrapped back into [0, 2*pi).
    _, _, nth = car.forward(ctrl_freq=10, current_state=[0.0, 0.0, -0.05], u=0.0)
    assert 0.0 <= nth < 2 * math.pi


def test_dubinscar_optCtrl_is_bang_bang():
    car = DubinsCar()  # uMode="min", wMax=1
    assert float(car.optCtrl_inPython([0.0, 0.0, 1.0])) == pytest.approx(-1.0)
    assert float(car.optCtrl_inPython([0.0, 0.0, -1.0])) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
#  Dubins car (3-D, speed + turn-rate control)
# --------------------------------------------------------------------------- #
def test_dubinscar2_dynamics_inPython():
    car = DubinsCar2()  # uMode="min"
    dx, dy, dtheta = car.dynamics_inPython(state=[0.0, 0.0, 0.0], control=[0.8, 0.5])
    assert (dx, dy, dtheta) == pytest.approx((0.8, 0.0, 0.5))


def test_dubinscar2_forward():
    car = DubinsCar2()
    nx, ny, nth = car.forward(ctrl_freq=10, current_state=[0.0, 0.0, 0.0],
                              control=[0.8, 0.0])
    assert (nx, ny, nth) == pytest.approx((0.08, 0.0, 0.0))


def test_dubinscar2_optCtrl_is_bang_bang():
    car = DubinsCar2()  # uMode="min"; speed in [-0.1, 0.8], w in [-1, 1]
    # coefficient = d0*cos(theta) + d1*sin(theta); with theta=0 it equals d0.
    np.testing.assert_allclose(
        car.optCtrl_inPython(state=[0.0, 0.0, 0.0], spat_deriv=[1.0, 0.0, 1.0]),
        [-0.1, -1.0])
    np.testing.assert_allclose(
        car.optCtrl_inPython(state=[0.0, 0.0, 0.0], spat_deriv=[-1.0, 0.0, -1.0]),
        [0.8, 1.0])


# --------------------------------------------------------------------------- #
#  Dubins car (4-D, acceleration + turn-rate control)
# --------------------------------------------------------------------------- #
def test_dubinscar4d_dynamics_inPython():
    car = DubinsCar4D()  # uMode="min"
    # state = [x, y, v, theta]
    x_dot, y_dot, v_dot, th_dot = car.dynamics_inPython(state=[0, 0, 2.0, 0.0],
                                                        action=[0.5, 0.3])
    assert (x_dot, y_dot, v_dot, th_dot) == pytest.approx((2.0, 0.0, 0.5, 0.3))


def test_dubinscar4d_forward():
    car = DubinsCar4D()
    nx, ny, nv, nth = car.forward(ctrl_freq=10, current_state=[0, 0, 1.0, 0.0],
                                  action=[0.0, 0.0])
    assert (nx, ny, nv, nth) == pytest.approx((0.1, 0.0, 1.0, 0.0))


def test_dubinscar4d_optCtrl_is_bang_bang():
    car = DubinsCar4D()  # uMode="min", uMax=[1,1], uMin=[-1,-1]
    assert car.optCtrl_inPython([0, 0, 1.0, 1.0]) == pytest.approx((-1.0, -1.0))
    assert car.optCtrl_inPython([0, 0, -1.0, -1.0]) == pytest.approx((1.0, 1.0))


# --------------------------------------------------------------------------- #
#  Dubins car (4-D, kinematic-bicycle variant) -- optCtrl helper only
# --------------------------------------------------------------------------- #
def test_dubinscar4d2_optCtrl_is_bang_bang():
    car = DubinsCar4D2()  # uMode="max", uMax=[1.5, pi/18], uMin=[-1.5, -pi/18]
    a, w = car.optCtrl_inPython([0, 0, -1.0, -1.0])
    assert (a, w) == pytest.approx((-1.5, -math.pi / 18))
    a, w = car.optCtrl_inPython([0, 0, 1.0, 1.0])
    assert (a, w) == pytest.approx((1.5, math.pi / 18))


# --------------------------------------------------------------------------- #
#  Dubins airplane (6-D)
# --------------------------------------------------------------------------- #
def test_dubinsairplane6d_dynamics_inPython():
    plane = DubinsAirplane6D()  # v=1, g=9.81, uMode="min"
    # state = [x, y, z, psi, gamma, phi]; action = [a, u_gamma, u_phi]
    xd, yd, zd, psid, gd, phid = plane.dynamics_inPython(
        state=[0, 0, 0, 0.0, 0.0, 0.0], action=[0.0, 0.2, 0.3])
    assert (xd, yd, zd) == pytest.approx((1.0, 0.0, 0.0))
    assert psid == pytest.approx(0.0)     # g/v * tan(phi=0)
    assert (gd, phid) == pytest.approx((0.2, 0.3))


def test_dubinsairplane6d_forward_wraps_angles():
    plane = DubinsAirplane6D()
    state = plane.forward(ctrl_freq=10, current_state=[0, 0, 0, 0, 0, 0],
                          action=[0, 0, 0])
    assert state[0] == pytest.approx(0.1)  # x advances by v*cos*cos*dt
    for angle in state[3:]:                # psi, gamma, phi all wrapped
        assert 0.0 <= angle < 2 * math.pi


def test_dubinsairplane6d_optCtrl_is_bang_bang():
    """Also a regression guard for the ``self.uMin[2]`` IndexError: the
    ``spat_deriv[5] > 0`` branch (below) used to index past the 2-element
    ``uMin`` list."""
    plane = DubinsAirplane6D()  # uMode="min", uMax=[1,1], uMin=[-1,-1]
    u_gamma, u_phi = plane.optCtrl_inPython([0, 0, 0, 0, 1.0, 1.0])
    assert (u_gamma, u_phi) == pytest.approx((-1.0, -1.0))
    u_gamma, u_phi = plane.optCtrl_inPython([0, 0, 0, 0, -1.0, -1.0])
    assert (u_gamma, u_phi) == pytest.approx((1.0, 1.0))


# --------------------------------------------------------------------------- #
#  Dubins capture (pursuit-evasion differential game) -- optCtrl helper only
# --------------------------------------------------------------------------- #
def test_dubinscapture_optCtrl_is_bang_bang():
    game = DubinsCapture()  # uMode="max", wMax=1
    # a_term = d0*x1 - d1*x0 - d2; with state at the origin a_term = -d2.
    assert game.optCtrl_inPython([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]) == pytest.approx(1.0)
    assert game.optCtrl_inPython([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]) == pytest.approx(-1.0)
