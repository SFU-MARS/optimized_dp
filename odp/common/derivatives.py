import numpy as np
from typing import List, Tuple, Union

try:
    from odp.Grid import Grid
except ImportError:
    pass


def _slice_dim(arr: np.ndarray, dim: int, start: int, length: int) -> np.ndarray:
    """Helper to extract a slice of fixed length along a given axis."""
    sl = [slice(None)] * arr.ndim
    sl[dim] = slice(start, start + length)
    return arr[tuple(sl)]


# =============================================================================
# Ghost Cell Boundary Conditions
# =============================================================================

def addGhostPeriodic(data: np.ndarray, dim: int, ghost_cells: int) -> np.ndarray:
    """Wrap boundary data into ghost cells for periodic dimensions.

    Mirrors toolboxLS's addGhostPeriodic: ghost cells on the low side are
    filled with the top of the array, ghost cells on the high side are
    filled with the bottom of the array.
    """
    pad_width = [(0, 0)] * data.ndim
    pad_width[dim] = (ghost_cells, ghost_cells)
    return np.pad(data, pad_width, mode="wrap")


def addGhostExtrapolate(data: np.ndarray, dim: int, num_ghost_cells: int) -> np.ndarray:
    """Linearly extrapolate boundary data into ghost cells."""
    idx_0 = [slice(None)] * data.ndim
    idx_1 = [slice(None)] * data.ndim
    idx_last = [slice(None)] * data.ndim
    idx_second_last = [slice(None)] * data.ndim

    idx_0[dim] = slice(0, 1)
    idx_1[dim] = slice(1, 2)
    idx_last[dim] = slice(-1, None)
    idx_second_last[dim] = slice(-2, -1)

    # Left ghost cells
    diff_bot = data[tuple(idx_0)] - data[tuple(idx_1)]
    slope_bot = (1) * abs(diff_bot) * np.sign(data[tuple(idx_0)]) 

    left_ghosts = [data[tuple(idx_0)] + width * slope_bot for width in range(num_ghost_cells, 0, -1)]
    left_pad = np.concatenate(left_ghosts, axis=dim)

    # Right ghost cells
    diff_top = data[tuple(idx_last)] - data[tuple(idx_second_last)]
    slope_top = (1) * abs(diff_top) * np.sign(data[tuple(idx_last)])
    right_ghosts = [data[tuple(idx_last)] + width * slope_top for width in range(1, num_ghost_cells + 1)]
    right_pad = np.concatenate(right_ghosts, axis=dim)

    return np.concatenate([left_pad, data, right_pad], axis=dim)


def addGhost(grid, data: np.ndarray, dim: int, ghost_cells: int) -> np.ndarray:
    """Add ghost cells using periodic or linear extrapolation based on grid.pDim."""
    if dim in grid.pDim:
        return addGhostPeriodic(data, dim, ghost_cells)
    return addGhostExtrapolate(data, dim, ghost_cells)


# =============================================================================
# Spatial Derivative Schemes
# =============================================================================

def upwindFirstFirst(
    grid, data: np.ndarray, dim: int = None, generateAll: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Computes 1st order upwind spatial derivatives (left and right approximations).

    Args:
        grid: odp.Grid instance
        data: ndarray of values at grid points
        dim: dimension along which to compute derivative
        generateAll: if True, computes derivatives for all dimensions
    """
    if dim is None or generateAll:
        derivL, derivR = [], []
        for d in range(grid.dims):
            dL, dR = upwindFirstFirst(grid, data, dim=d, generateAll=False)
            derivL.append(dL)
            derivR.append(dR)
        return derivL, derivR

    dx = grid.dx[dim]
    n_pts = data.shape[dim]

    # 1 ghost cell per side required for 1st order
    padded = addGhost(grid, data, dim, ghost_cells=1)

    # Left derivative (backward difference): (phi_i - phi_{i-1}) / dx
    derivL = (_slice_dim(padded, dim, 1, n_pts) - _slice_dim(padded, dim, 0, n_pts)) / dx

    # Right derivative (forward difference): (phi_{i+1} - phi_i) / dx
    derivR = (_slice_dim(padded, dim, 2, n_pts) - _slice_dim(padded, dim, 1, n_pts)) / dx

    return derivL, derivR


def _upwindFirstENO2(grid, data: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """2nd order ENO directional spatial derivative approximation."""
    dx = grid.dx[dim]
    n_pts = data.shape[dim]

    # 2 ghost cells per side
    padded = addGhost(grid, data, dim, ghost_cells=2)

    # Divided differences
    # D1 length: n_pts + 3
    D1 = (_slice_dim(padded, dim, 1, n_pts + 3) - _slice_dim(padded, dim, 0, n_pts + 3)) / dx
    # D2 length: n_pts + 2
    D2 = (_slice_dim(D1, dim, 1, n_pts + 2) - _slice_dim(D1, dim, 0, n_pts + 2)) / (2.0 * dx)

    # --- Left derivative (backward) ---
    D1_L = _slice_dim(D1, dim, 1, n_pts)
    D2_L = _slice_dim(D2, dim, 0, n_pts)
    D2_R = _slice_dim(D2, dim, 1, n_pts)

    # Pick smaller magnitude; tie favors left stencil
    chosen_D2_L = np.where(np.abs(D2_L) <= np.abs(D2_R), D2_L, D2_R)
    derivL = D1_L + dx * chosen_D2_L

    # --- Right derivative (forward) ---
    D1_R = _slice_dim(D1, dim, 2, n_pts)
    D2_L = _slice_dim(D2, dim, 1, n_pts)
    D2_R = _slice_dim(D2, dim, 2, n_pts)

    # Pick smaller magnitude; tie favors left stencil
    chosen_D2_R = np.where(np.abs(D2_R) < np.abs(D2_L), D2_R, D2_L)
    derivR = D1_R - dx * chosen_D2_R

    return derivL, derivR


def upwindFirstENO(
    grid, data: np.ndarray, dim: int = None, order: int = 2, generateAll: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Computes ENO directional spatial derivatives (order 1, 2, or 3).
    """
    if dim is None or generateAll:
        derivL, derivR = [], []
        for d in range(grid.dims):
            dL, dR = upwindFirstENO(grid, data, dim=d, order=order, generateAll=False)
            derivL.append(dL)
            derivR.append(dR)
        return derivL, derivR

    if order == 1:
        return upwindFirstFirst(grid, data, dim=dim, generateAll=False)
    elif order == 2:
        return _upwindFirstENO2(grid, data, dim=dim)
    else:
        raise ValueError(f"Unsupported ENO order: {order}. Must be 1, 2, or 3.")


def upwindFirstENO2(grid, data: np.ndarray, dim: int = None, generateAll: bool = False):
    return upwindFirstENO(grid, data, dim=dim, order=2, generateAll=generateAll)
