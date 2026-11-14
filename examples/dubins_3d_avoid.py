import imp
import numpy as np
# Utility functions to initialize the problem1
from odp.Grid import Grid
from odp.Shapes import *
from pathlib import Path
import shutil


# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar
# Plot options
from odp.Plots import PlotOptions, visualize_plots

import matplotlib
matplotlib.use("Agg")  # headless backend (docker): render straight to files, no display needed
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import math

# Define grid
g = Grid(np.array([-2.0, -2.0, -math.pi]), np.array([2.0, 2.0, math.pi]), np.array([60, 60, 36]), [2])

# Define my object
my_car = DubinsCar(uMode="max", dMode="min")
obstacle_center = np.array([0.0, 0.0, 0.0])
obstacle_radius = 0.8

# Use the grid to initialize initial value function
Initial_value_f = CylinderShape(g, [2], np.array([0., 0., 0.]), 0.8)

# Look-back length and time step
lookback_length = 1.5
t_step = 0.05

small_number = 1e-5

tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# In this example, we compute a Backward Reachable Tube
compMethods = { "TargetSetMode": "minVWithV0"}
result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, saveAllTimeSteps=True, accuracy="medium")

last_time_step_result = result[..., 0]


# ======================================================================
# VISUALIZATION  (saves files only; nothing is displayed)
# ======================================================================
theta_idx = 0  # which slice of the heading axis to plot
fps = 10
hold_last_frames = 10  # freeze on the final frame for ~1 s at the end of the video
 
out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)
stem = f"dubins_brt_theta_idx{theta_idx}"
 
x, y = g.grid_points[0], g.grid_points[1]
theta = g.grid_points[2][theta_idx]
X, Y = np.meshgrid(x, y, indexing="ij")
 
# Time axis of `result`: index -1 is t = 0 (initial V0), index 0 is t = lookback_length.
n_t = result.shape[-1]
times = tau[::-1]  # times[k] is the look-back time belonging to result[..., k]
V_all = result[:, :, theta_idx, :]  # (60, 60, n_t), indexed [x, y, time]
 
# One fixed colour scale for every frame, centred exactly on V = 0
vmin, vmax = min(V_all.min(), -1e-6), max(V_all.max(), 1e-6)
levels = np.concatenate([np.linspace(vmin, 0.0, 13), np.linspace(0.0, vmax, 13)[1:]])
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
cmap = plt.get_cmap("RdBu_r")
 
OBSTACLE_FACE = (0.12, 0.12, 0.12, 0.55)
 
fig, ax = plt.subplots(figsize=(7.8, 6.6), constrained_layout=True)  # 1170x990 px at 150 dpi (even, as h264 needs)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, shrink=0.9)
cbar.set_label(r"$V(x,\,y,\,\theta)$", fontsize=12)
cbar.ax.axhline(0.0, color="black", linewidth=1.8)
 
 
def draw_slice(V, tau_now):
    """Redraw the (x, y) slice V on the shared axes for look-back time tau_now."""
    ax.clear()
 
    # Filled value function + zero level set (boundary of the backward reachable tube)
    ax.contourf(X, Y, V, levels=levels, cmap=cmap, norm=norm, zorder=1)
    ax.contour(X, Y, V, levels=[0.0], colors="black", linewidths=2.6, zorder=4)
 
    # Obstacle (semi-transparent fill + hatch so the value function stays visible underneath)
    ax.add_patch(
        Circle(
            (obstacle_center[0], obstacle_center[1]),
            obstacle_radius,
            facecolor=OBSTACLE_FACE,
            edgecolor="black",
            linewidth=2.0,
            hatch="//",
            zorder=5,
        )
    )
    ax.text(
        obstacle_center[0], obstacle_center[1], "Obstacle",
        ha="center", va="center", color="white", fontsize=12, fontweight="bold", zorder=6,
    )
 
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_axisbelow(False)
    ax.grid(True, color="white", alpha=0.25, linewidth=0.6)
    ax.set_title(
        rf"Value function slice at $\theta = {theta:.2f}$ rad ({np.degrees(theta):.0f}°), index {theta_idx}"
        "\n"
        rf"look-back time $\tau = {tau_now:.2f}$ / {lookback_length}",
        fontsize=12.5, loc="left",
    )
    ax.legend(
        handles=[
            Patch(facecolor=OBSTACLE_FACE, edgecolor="black", hatch="//",
                  label=f"Obstacle (r = {obstacle_radius})"),
            Line2D([0], [0], color="black", linewidth=2.6, label=r"$V = 0$ (BRT boundary)"),
        ],
        loc="upper right", framealpha=0.92, fontsize=10,
    )
 
 
# ---- 1) Still image: final look-back time (result[..., 0]) -------------------------
draw_slice(V_all[..., 0], times[0])
for ext in ("png", "pdf"):
    fig.savefig(out_dir / f"{stem}_final.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved still image: {(out_dir / f'{stem}_final.png').resolve()}")
 
 
# ---- 2) Animation: BRT growing from t = 0 to t = lookback_length -------------------
def find_ffmpeg():
    """System ffmpeg if present, else the binary bundled with `pip install imageio-ffmpeg`."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None
 
 
ffmpeg = find_ffmpeg()
if ffmpeg:
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=4000,
        extra_args=["-pix_fmt", "yuv420p"],  # plays in browsers / QuickTime / VLC
    )
    video_path = out_dir / f"{stem}.mp4"
else:
    print("ffmpeg not found (try `apt-get install -y ffmpeg` or `pip install imageio-ffmpeg`); "
          "falling back to GIF.")
    writer = animation.PillowWriter(fps=fps)
    video_path = out_dir / f"{stem}.gif"
 
# Frames run from the initial value function (index -1) to the final one (index 0)
frame_ids = list(range(n_t - 1, -1, -1)) + [0] * hold_last_frames
 
with writer.saving(fig, str(video_path), dpi=150):
    for i, k in enumerate(frame_ids):
        draw_slice(V_all[..., k], times[k])
        writer.grab_frame()
        if i % 10 == 0:
            print(f"  rendered frame {i + 1}/{len(frame_ids)}")
 
print(f"Saved animation: {video_path.resolve()}")
plt.close(fig)