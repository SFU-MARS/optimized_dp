"""
Reach-avoid HJ game for a 3D Dubins car (v, w controls), aligned with ToyDubinsCarNavEnv.

Reach : enter the goal disk (radius goal_size) centered at goal_pos.
Avoid : enter the obstacle disk (radius agent_radius + hazards_size) centered at obs_pos.
Disturbances are zero (dMax = 0).

Map (from dubins_car.py defaults):
    map_size      = 5.0   -> x, y in [-2.5, 2.5]
    goal_pos      = [1.5, 1.5],  goal_size   = 0.3
    obs_pos       = [0.0, 0.0],  hazards_size = 0.6
    agent_radius  = 0.1   -> effective obstacle radius = 0.7
    v_max = 1.0, w_max = 1.0
"""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from odp.Grid import Grid
from odp.Plots import PlotOptions, plot_isosurface
from odp.Shapes import CylinderShape
from odp.dynamics.DubinsCar3D2Ctrls import DubinsCar3D2Ctrls
from odp.solver import HJSolver

# ---------------------------------------------------------------------------
# Map / task parameters (ToyDubinsCarNavEnv)
# ---------------------------------------------------------------------------
MAP_SIZE = 5.0
GOAL_POS = np.array([1.5, 1.5], dtype=np.float64)
OBS_POS = np.array([0.0, 0.0], dtype=np.float64)
HAZARDS_SIZE = 0.6
GOAL_SIZE = 0.3
AGENT_RADIUS = 0.1

V_MAX = 0.5
W_MAX = 0.2

MAP_HALF = MAP_SIZE / 2.0
OBSTACLE_RADIUS = AGENT_RADIUS + HAZARDS_SIZE  # 0.7, inflated obstacle

# ---------------------------------------------------------------------------
# Solver settings (tune grid_size / lookback_length for accuracy vs runtime)
# ---------------------------------------------------------------------------
GRID_SIZE = 100
LOOKBACK_LENGTH = 12.0
T_STEP = 0.05
ACCURACY = "medium"

SAVE_DIR = "values"
PLOT_DIR = "plots"

# Heading slice for BRT plot (diagonal toward goal at origin)
THETA_BRT = math.pi / 4
# Set True to also open the legacy Plotly isosurface in a browser
SAVE_PLOTLY_BRT = False
# If True, recompute even when the .npy cache exists
FORCE_RECOMPUTE = False
# Save full V(x,y,theta,tau) from HJSolver; required for BRT evolution plots.
SAVE_ALL_TIME_STEPS = True
# Number of panels in the evolution figure (initial -> final backward time)
N_EVOLUTION_SNAPSHOTS = 8
# Also write an animated GIF of BRT growth (needs pillow for .gif)
SAVE_BRT_EVOLUTION_GIF = True


def config_tag():
    """Shared tag for .npy / plot filenames (change V_MAX, W_MAX -> new cache)."""
    return (
        f"g{GRID_SIZE}_map{MAP_SIZE}_v{V_MAX:g}_w{W_MAX:g}_"
        f"goal{GOAL_SIZE}_obs{OBSTACLE_RADIUS}"
    )


def save_name():
    suffix = "_alltau" if SAVE_ALL_TIME_STEPS else ""
    return f"dubins_car_reach_avoid_{config_tag()}{suffix}.npy"


def plot_basename():
    return f"dubins_reach_avoid_{config_tag()}_theta_pi4"


def theta_to_slice_index(grid, theta):
    """Nearest grid index along the periodic heading dimension."""
    theta_vals = grid.grid_points[2]
    return int(np.argmin(np.abs(theta_vals - theta)))


def make_initial_value(grid):
    """init_value = max(reach, -avoid) used inside HJSolver at t=0 (backward time)."""
    reach_set = CylinderShape(
        grid,
        ignore_dims=[2],
        center=[GOAL_POS[0], GOAL_POS[1], 0.0],
        radius=GOAL_SIZE,
    )
    avoid_set = CylinderShape(
        grid,
        ignore_dims=[2],
        center=[OBS_POS[0], OBS_POS[1], 0.0],
        radius=OBSTACLE_RADIUS,
    )
    return np.maximum(reach_set, -avoid_set)


def backward_time_for_slice(slice_idx, tau):
    """
    Map last-axis index to backward time (odp/solver.py saveAllTimeSteps layout).

    valfuncs[..., 0]     = V at tau[-1] (final horizon T)
    valfuncs[..., -1]    = V at tau[0]  (initial / terminal condition)
    """
    return float(tau[-1 - slice_idx])


def load_tau(meta_path):
    if os.path.isfile(meta_path):
        meta = np.load(meta_path)
        if "tau" in meta:
            return meta["tau"]
    small_number = 1e-5
    return np.arange(
        start=0.0, stop=LOOKBACK_LENGTH + small_number, step=T_STEP
    )


def extract_final_value(V, grid, meta_path=None):
    """
    Return spatial value function used for plotting.

    HJSolver with saveAllTimeSteps=False returns final V with shape (nx, ny, ntheta).
    If a 4D array was saved (..., n_tau), take index 0 (final backward time); see
    odp/solver.py: valfuncs[..., 0] = V_t after the loop.
    """
    spatial_shape = tuple(grid.pts_each_dim)
    if V.ndim == len(spatial_shape) and tuple(V.shape) == spatial_shape:
        return V, {"kind": "final_spatial", "time_index": None}

    if V.ndim == len(spatial_shape) + 1:
        n_tau = V.shape[-1]
        if meta_path and os.path.isfile(meta_path):
            meta = np.load(meta_path)
            if "final_time_index" in meta:
                t_idx = int(meta["final_time_index"])
            else:
                t_idx = 0
        else:
            t_idx = 0
        print(
            f"  Detected 4D value array (..., {n_tau}); using time index {t_idx} "
            f"as final backward-time slice (index -1 is initial)."
        )
        return V[..., t_idx], {
            "kind": "time_series",
            "time_index": t_idx,
            "n_tau": n_tau,
        }

    raise ValueError(
        f"Unexpected value shape {V.shape}; expected {spatial_shape} or "
        f"{spatial_shape + ('T',)}."
    )


def diagnose_value_function(V_final, V_init, lookback_length):
    """Print checks that V_final is post-solve, not the initial condition."""
    diff = np.abs(V_final - V_init)
    frac_final = float((V_final <= 0).mean())
    frac_init = float((V_init <= 0).mean())

    print("-" * 60)
    print("Value function check (plot uses FINAL backward-time V, not V0)")
    print(f"  V shape (plotted) : {V_final.shape}  (spatial: x, y, theta)")
    print(f"  V0 shape (init)     : {V_init.shape}")
    print(f"  max |V - V0|        : {diff.max():.4f}")
    print(f"  mean |V - V0|       : {diff.mean():.4f}")
    print(f"  fraction V  <= 0    : {frac_final:.2%}  (BRT interior at T={lookback_length}s)")
    print(f"  fraction V0 <= 0    : {frac_init:.2%}  (goal set only at init)")
    if np.allclose(V_final, V_init):
        print("  WARNING: V equals V0 — solver may not have run or wrong slice loaded!")
    else:
        print("  OK: V differs from V0 — using integrated value function.")
    print("-" * 60)


def _draw_map_static(ax):
    """Goal, obstacle, and map border (shared by scene and evolution plots)."""
    ax.add_patch(
        Rectangle(
            (-MAP_HALF, -MAP_HALF),
            MAP_SIZE,
            MAP_SIZE,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
        )
    )
    ax.add_patch(
        Circle(
            OBS_POS,
            OBSTACLE_RADIUS,
            facecolor="tab:red",
            edgecolor="darkred",
            alpha=0.40,
            linewidth=1.0,
            zorder=4,
        )
    )
    ax.add_patch(
        Circle(
            OBS_POS,
            HAZARDS_SIZE,
            fill=False,
            edgecolor="tab:red",
            linewidth=0.8,
            linestyle="--",
            zorder=4,
        )
    )
    ax.add_patch(
        Circle(
            GOAL_POS,
            GOAL_SIZE,
            facecolor="tab:green",
            edgecolor="darkgreen",
            alpha=0.40,
            linewidth=1.0,
            zorder=4,
        )
    )
    ax.set_xlim(-MAP_HALF, MAP_HALF)
    ax.set_ylim(-MAP_HALF, MAP_HALF)
    ax.set_aspect("equal")


def _draw_brt_on_ax(ax, X, Y, V_xy, vmax):
    """Draw V<=0 fill and V=0 contour on one axes."""
    ax.contourf(
        X, Y, V_xy,
        levels=[-vmax, 0.0],
        colors=["#FF00FF"],
        alpha=0.20,
        zorder=7,
    )
    ax.contour(
        X, Y, V_xy,
        levels=[0.0],
        colors="white",
        linewidths=3.0,
        zorder=9,
    )
    brt_line = ax.contour(
        X, Y, V_xy,
        levels=[0.0],
        colors="#CC00CC",
        linewidths=2.0,
        zorder=10,
    )
    return brt_line


def plot_reach_avoid_scene(
    grid,
    value_fn,
    theta_idx,
    theta_rad,
    save_path,
    show=False,
    lookback_length=LOOKBACK_LENGTH,
    time_label=None,
):
    """
    Custom 2D scene: map bounds, goal, obstacle, and BRT (V=0) at fixed heading.

    value_fn must be spatial V with shape (nx, ny, ntheta).
    """
    x = grid.grid_points[0]
    y = grid.grid_points[1]
    V_xy = value_fn[:, :, theta_idx]
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, ax = plt.subplots(figsize=(8, 8))

    vmax = np.percentile(np.abs(V_xy), 99)
    vmax = max(vmax, 0.5)
    cf = ax.contourf(
        X, Y, V_xy,
        levels=25,
        cmap="RdYlGn",
        alpha=0.30,
        vmin=-vmax,
        vmax=vmax,
    )
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="V(x, y, θ)")

    _draw_map_static(ax)
    _draw_brt_on_ax(ax, X, Y, V_xy, vmax)

    arrow_len = 0.55
    ax.annotate(
        "",
        xy=(arrow_len * math.cos(theta_rad), arrow_len * math.sin(theta_rad)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="navy", lw=2.0),
        zorder=6,
    )

    title_time = (
        time_label
        if time_label is not None
        else f"backward T = {lookback_length:.1f} s (final)"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Dubins Reach-Avoid — {title_time}\n"
        f"θ = {theta_rad:.3f} rad | magenta: V=0, fill: V≤0"
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_brt_evolution(
    V_series,
    grid,
    tau,
    theta_idx,
    theta_rad,
    snapshot_path,
    gif_path=None,
    n_snapshots=N_EVOLUTION_SNAPSHOTS,
):
    """
    Plot how the BRT slice grows over backward time.

    V_series shape (nx, ny, ntheta, n_tau); index 0 = tau[-1], index -1 = tau[0].
    """
    x = grid.grid_points[0]
    y = grid.grid_points[1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    n_tau = V_series.shape[-1]

    # Global color scale from all slices at this heading
    all_slices = V_series[:, :, theta_idx, :]
    vmax = np.percentile(np.abs(all_slices), 99)
    vmax = max(float(vmax), 0.5)

    n_snapshots = min(n_snapshots, n_tau)
    # initial (index n_tau-1) -> final (index 0)
    slice_indices = np.linspace(n_tau - 1, 0, n_snapshots, dtype=int)

    ncols = min(4, n_snapshots)
    nrows = int(math.ceil(n_snapshots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    print("-" * 60)
    print("BRT evolution (backward time increases left -> right, top -> bottom)")
    for panel, t_idx in enumerate(slice_indices):
        ax = axes[panel]
        V_xy = V_series[:, :, theta_idx, t_idx]
        t_back = backward_time_for_slice(t_idx, tau)
        frac_neg = float((V_xy <= 0).mean())

        ax.contourf(
            X, Y, V_xy,
            levels=25,
            cmap="RdYlGn",
            alpha=0.35,
            vmin=-vmax,
            vmax=vmax,
        )
        _draw_map_static(ax)
        _draw_brt_on_ax(ax, X, Y, V_xy, vmax)
        ax.set_title(
            f"t_back = {t_back:.2f} s\nidx={t_idx}, V≤0: {frac_neg:.1%}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.2)
        print(f"  panel {panel}: slice idx={t_idx}, t_back={t_back:.2f}s, V<=0: {frac_neg:.1%}")

    for j in range(len(slice_indices), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"BRT growth (θ = {theta_rad:.3f} rad) — backward reachable tube",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(snapshot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evolution snapshots to {snapshot_path}")

    if gif_path is None or not SAVE_BRT_EVOLUTION_GIF:
        return

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Skipping GIF (matplotlib animation not available).")
        return

    # Animate every N-th slice to keep file size reasonable
    step = max(1, n_tau // 60)
    frame_indices = list(range(n_tau - 1, -1, -step))
    if frame_indices[-1] != 0:
        frame_indices.append(0)

    fig_a, ax_a = plt.subplots(figsize=(7, 7))

    def update(frame_num):
        t_idx = frame_indices[frame_num]
        V_xy = V_series[:, :, theta_idx, t_idx]
        t_back = backward_time_for_slice(t_idx, tau)
        ax_a.clear()
        ax_a.contourf(
            X, Y, V_xy,
            levels=25,
            cmap="RdYlGn",
            alpha=0.35,
            vmin=-vmax,
            vmax=vmax,
        )
        _draw_map_static(ax_a)
        _draw_brt_on_ax(ax_a, X, Y, V_xy, vmax)
        ax_a.set_title(
            f"backward t = {t_back:.2f} s  (slice {t_idx}/{n_tau - 1})\n"
            f"θ = {theta_rad:.3f} rad",
            fontsize=11,
        )
        ax_a.grid(True, alpha=0.2)
        return []

    anim = FuncAnimation(
        fig_a,
        update,
        frames=len(frame_indices),
        interval=120,
        blit=False,
    )
    anim.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig_a)
    print(f"Saved evolution GIF to {gif_path}")
    print("-" * 60)


def build_grid():
    grid_min = np.array([-MAP_HALF, -MAP_HALF, -math.pi])
    grid_max = np.array([MAP_HALF, MAP_HALF, math.pi])
    return Grid(
        grid_min,
        grid_max,
        dims=3,
        pts_each_dim=np.array([GRID_SIZE, GRID_SIZE, GRID_SIZE]),
        periodicDims=[2],
    )


def cache_paths():
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, save_name())
    meta_path = save_path.replace(".npy", "_meta.npz")
    return save_path, meta_path


def load_cached_value(save_path, grid, meta_path=None):
    raw = np.load(save_path)
    spatial_shape = tuple(grid.pts_each_dim)
    if raw.ndim == len(spatial_shape) and tuple(raw.shape) == spatial_shape:
        return raw
    if raw.ndim == len(spatial_shape) + 1 and tuple(raw.shape[:3]) == spatial_shape:
        V_final, _ = extract_final_value(raw, grid, meta_path)
        return V_final
    raise ValueError(
        f"Cached value shape {raw.shape} not compatible with grid {spatial_shape}. "
        "Delete the .npy or set FORCE_RECOMPUTE = True."
    )


def solve_and_save(grids, save_path, meta_path, theta_idx):
    dubins = DubinsCar3D2Ctrls(
        v_max=V_MAX,
        w_max=W_MAX,
        dt=0.1,
        dMax=(0.0, 0.0, 0.0),
        uMode="min",
        dMode="max",
    )

    reach_set = CylinderShape(
        grids,
        ignore_dims=[2],
        center=[GOAL_POS[0], GOAL_POS[1], 0.0],
        radius=GOAL_SIZE,
    )
    avoid_set = CylinderShape(
        grids,
        ignore_dims=[2],
        center=[OBS_POS[0], OBS_POS[1], 0.0],
        radius=OBSTACLE_RADIUS,
    )

    small_number = 1e-5
    tau = np.arange(start=0.0, stop=LOOKBACK_LENGTH + small_number, step=T_STEP)
    po = PlotOptions(
        do_plot=False,
        plot_type="set",
        plotDims=[0, 1],
        slicesCut=[theta_idx],
    )
    comp_methods = {
        "TargetSetMode": "minVWithVTarget",
        "ObstacleSetMode": "maxVWithObstacle",
    }

    print("=" * 60)
    print("Dubins car reach-avoid — computing value function")
    print(f"  grid        : {GRID_SIZE}^3  ->  dx ≈ {grids.dx[0]:.4f} m")
    print(f"  map         : [{-MAP_HALF}, {MAP_HALF}]^2")
    print(f"  goal        : center {GOAL_POS.tolist()}, radius {GOAL_SIZE}")
    print(f"  obstacle    : center {OBS_POS.tolist()}, radius {OBSTACLE_RADIUS}")
    print(f"  controls    : v in [0, {V_MAX}], w in [{-W_MAX}, {W_MAX}]")
    print(f"  disturbance : 0")
    print(f"  horizon     : {LOOKBACK_LENGTH} s, dt = {T_STEP}")
    print("=" * 60)

    solve_start_time = time.time()
    result = HJSolver(
        dubins,
        grids,
        [reach_set, avoid_set],
        tau,
        comp_methods,
        po,
        saveAllTimeSteps=SAVE_ALL_TIME_STEPS,
        accuracy=ACCURACY,
    )
    solve_end_time = time.time()

    process = psutil.Process(os.getpid())
    print(f"CPU memory during solve: {process.memory_info().rss / 1e9:.2f} GB")
    print(f"HJSolver return shape: {result.shape}")
    print(f"Value function size  : {result.nbytes / 1e9:.2f} GB")
    print(f"HJ solve time        : {solve_end_time - solve_start_time:.1f} s")

    if SAVE_ALL_TIME_STEPS:
        print(
            "  saveAllTimeSteps=True -> 4D array; final slice is [..., 0], "
            "initial is [..., -1]."
        )
    else:
        print("  saveAllTimeSteps=False -> 3D final V (nx, ny, theta) at backward time T.")

    np.save(save_path, result)
    final_time_index = 0 if SAVE_ALL_TIME_STEPS else -1
    np.savez(
        meta_path,
        grid_min=grids.min,
        grid_max=grids.max,
        pts_each_dim=grids.pts_each_dim,
        periodic_dims=np.array([2]),
        tau=tau,
        save_all_time_steps=SAVE_ALL_TIME_STEPS,
        final_time_index=final_time_index,
        initial_time_index=-1,
        time_index_note="[...,0]=tau[-1] final; [..., -1]=tau[0] initial",
        value_description="final_backward_value_at_lookback_T",
        map_size=MAP_SIZE,
        goal_pos=GOAL_POS,
        obs_pos=OBS_POS,
        goal_size=GOAL_SIZE,
        hazards_size=HAZARDS_SIZE,
        agent_radius=AGENT_RADIUS,
        obstacle_radius=OBSTACLE_RADIUS,
        v_max=V_MAX,
        w_max=W_MAX,
        lookback_length=LOOKBACK_LENGTH,
        t_step=T_STEP,
        u_mode="min",
        d_mode="max",
    )
    print(f"Saved value function to {save_path}")
    print(f"Saved metadata to       {meta_path}")
    return result


def visualize_result(grids, result, theta_idx, V_series=None, tau=None):
    theta_at_idx = grids.grid_points[2][theta_idx]
    os.makedirs(PLOT_DIR, exist_ok=True)
    scene_plot_path = os.path.join(PLOT_DIR, f"{plot_basename()}_scene.png")

    print("-" * 60)
    print("Final-scene visualization (matplotlib)")
    print(f"  requested theta : {THETA_BRT:.4f} rad (pi/4, toward goal)")
    print(f"  grid index      : {theta_idx} / {GRID_SIZE - 1}")
    print(f"  actual theta    : {theta_at_idx:.4f} rad")
    print(f"  saving to       : {scene_plot_path}")
    print("-" * 60)

    plot_reach_avoid_scene(
        grids,
        result,
        theta_idx,
        theta_at_idx,
        scene_plot_path,
        show=False,
        lookback_length=LOOKBACK_LENGTH,
        time_label=f"backward T = {LOOKBACK_LENGTH:.1f} s (final)",
    )
    print(f"Saved scene plot to {scene_plot_path}")

    if V_series is not None and tau is not None:
        evo_path = os.path.join(PLOT_DIR, f"{plot_basename()}_brt_evolution.png")
        gif_path = os.path.join(PLOT_DIR, f"{plot_basename()}_brt_evolution.gif")
        plot_brt_evolution(
            V_series,
            grids,
            tau,
            theta_idx,
            theta_at_idx,
            evo_path,
            gif_path=gif_path,
        )

    if SAVE_PLOTLY_BRT:
        brt_plot_base = os.path.join(PLOT_DIR, f"{plot_basename()}_plotly_brt")
        brt_po = PlotOptions(
            do_plot=True,
            plot_type="set",
            plotDims=[0, 1],
            slicesCut=[theta_idx],
            save_fig=True,
            filename=brt_plot_base,
            interactive_html=True,
        )
        plot_isosurface(grids, result, brt_po)


def main():
    start_time = time.time()
    grids = build_grid()
    theta_idx = theta_to_slice_index(grids, THETA_BRT)
    save_path, meta_path = cache_paths()
    V_init = make_initial_value(grids)

    use_cache = (
        os.path.isfile(save_path)
        and not FORCE_RECOMPUTE
    )
    raw = None

    if use_cache:
        print("=" * 60)
        print("Found cached value function — skipping HJ solve")
        print(f"  load from : {save_path}")
        print("=" * 60)
        try:
            raw = np.load(save_path)
            if SAVE_ALL_TIME_STEPS and raw.ndim != 4:
                raise ValueError(
                    f"Expected 4D cache with SAVE_ALL_TIME_STEPS=True, got {raw.shape}."
                )
            if not SAVE_ALL_TIME_STEPS and raw.ndim != 3:
                raise ValueError(
                    f"Expected 3D cache with SAVE_ALL_TIME_STEPS=False, got {raw.shape}."
                )
            result, _ = extract_final_value(raw, grids, meta_path)
        except ValueError as exc:
            print(f"Cache invalid ({exc}); recomputing.")
            use_cache = False

    if not use_cache:
        if FORCE_RECOMPUTE and os.path.isfile(save_path):
            print("FORCE_RECOMPUTE=True — recomputing value function.")
        raw = solve_and_save(grids, save_path, meta_path, theta_idx)
        result, _ = extract_final_value(raw, grids, meta_path)

    tau = load_tau(meta_path)
    V_series = raw if raw.ndim == 4 else None
    if V_series is not None:
        print(f"Loaded time series: shape {V_series.shape} (x, y, θ, τ)")
        print(f"  slice [..., 0]  -> t_back = {backward_time_for_slice(0, tau):.2f} s (final)")
        print(
            f"  slice [..., -1] -> t_back = {backward_time_for_slice(V_series.shape[-1] - 1, tau):.2f} s (initial)"
        )

    diagnose_value_function(result, V_init, LOOKBACK_LENGTH)
    visualize_result(grids, result, theta_idx, V_series=V_series, tau=tau)
    print(f"Total wall time         : {time.time() - start_time:.1f} s")


if __name__ == "__main__":
    main()
