import pytest
import numpy as np

# Assuming your Grid class is in odp.Grid and derivatives are in spat_derivs.py
from odp.Grid import Grid
from odp.Shapes import *

from odp.common.derivatives import upwindFirstFirst, upwindFirstENO2
from odp.solver import computeSpatDerivArray


# =============================================================================
# 3D Test
# =============================================================================

def test_3d_against_analytical():
    """
    Validates ENO2 derivatives using the 3D grid setup from the 
    pursuit_evasion_game.py example.
    """
    # Setup Grid
    grid = Grid(
        minBounds=[-2., -2., -np.pi],
        maxBounds=[2., 2., np.pi],
        dims=3,
        pts_each_dim=[150, 150, 72],
        periodicDims=[2]
    )

    # Create a smooth analytical test function (V = x^2 + y^2 + sin(theta))
    # We use grid.vs to leverage broadcasting
    V = grid.vs[0]**2 + grid.vs[1]**2 + np.sin(grid.vs[2])

    # Define Analytical Derivatives
    analytic_dV = [
        np.broadcast_to(2 * grid.vs[0], V.shape),              # dV/dx
        np.broadcast_to(2 * grid.vs[1], V.shape),              # dV/dy
        np.broadcast_to(np.cos(grid.vs[2]), V.shape)           # dV/dtheta
    ]
    
    # Compute 1st Order Numerical Derivatives
    derivL, derivR = upwindFirstFirst(grid, V, dim=3, generateAll=True)
    for dim in range(grid.dims):
        center_deriv = (derivL[dim] + derivR[dim])/2 
        np.testing.assert_allclose(
            center_deriv, 
            analytic_dV[dim], 
            atol=3e-2, 
            err_msg=f"Centering derivative (1st ENO) failed for 3D PE game on dimension {dim}"
        )

    # Compute 2nd Order Numerical Derivatives
    derivL, derivR = upwindFirstENO2(grid, V, generateAll=True)

    # Assert Accuracy
    for dim in range(grid.dims):
        center_deriv = (derivL[dim] + derivR[dim])/2 
        np.testing.assert_allclose(
            center_deriv, 
            analytic_dV[dim], 
            atol=3e-2, 
            err_msg=f"Centering derivative (2nd ENO) failed for 3D PE game on dimension {dim}"
        )
        
# =============================================================================
# Comparison with computeSpatDerivArray
# =============================================================================

@pytest.mark.parametrize("dims, pts_per_dim", [
    (1, 100),  # 100^1 = 100 points
    (2, 250),  # 250^2 = 62,500 points
    (3, 40),  # 40^3 = 64,000 points
    (4, 15),  # 15^4 = 50,625 points
    (5, 11),  # 11^5 = 161,051 points
    (6, 8),   # 8^6 = 262,144 points
    (7, 5)    # 5^7 = 78,125 points
])
def test_against_computeSpatDerivArray(dims, pts_per_dim):
    """Ensure the Python ENO schemes match the backend computeSpatDerivArray across N-dimensions."""
    
    # 1. Setup Grid
    # Using the last dimension as periodic to test both boundary conditions
    grid = Grid(
        minBounds=[-2.0] * dims,
        maxBounds=[2.0] * dims,
        dims=dims,
        pts_each_dim=[pts_per_dim] * dims,
        periodicDims=[dims - 1] 
    )

    # 2. Generate Random Data
    V = np.random.rand(*grid.pts_each_dim)

    # 3. Test 1st Order ENO Derivatives (Accuracy: "low")
    derivL_1, derivR_1 = upwindFirstFirst(grid, V, generateAll=True)
    center_deriv_1 = [(derivL_1[i] + derivR_1[i]) / 2.0 for i in range(grid.dims)]

    for i in range(grid.dims):
        # Note: computeSpatDerivArray usually uses 1-based indexing for deriv_dim (i+1)
        heteroCL_odp_center_1 = computeSpatDerivArray(grid, V, deriv_dim=i + 1, accuracy="low")
        
        np.testing.assert_allclose(
            center_deriv_1[i], 
            heteroCL_odp_center_1,
            atol=1e-5,
            err_msg=f"1st ENO (low accuracy) failed for {dims}D grid on dim {i}"
        )

    # 4. Test 2nd Order ENO Derivatives (Accuracy: "medium")
    derivL_2, derivR_2 = upwindFirstENO2(grid, V, generateAll=True)
    center_deriv_2 = [(derivL_2[i] + derivR_2[i]) / 2.0 for i in range(grid.dims)]

    for i in range(grid.dims):
        heteroCL_odp_center_2 = computeSpatDerivArray(grid, V, deriv_dim=i + 1, accuracy="medium")
        
        np.testing.assert_allclose(
            center_deriv_2[i], 
            heteroCL_odp_center_2,
            atol=1e-5,
            err_msg=f"2nd ENO (medium accuracy) failed for {dims}D grid on dim {i}"
        )
