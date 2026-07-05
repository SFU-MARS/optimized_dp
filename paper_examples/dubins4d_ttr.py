import os
import time
import psutil

import numpy as np

from odp.Grid import Grid
from odp.solver import TTRSolver
from odp.dynamics import DubinsCar4D
from odp.Plots import PlotOptions
from odp.Shapes import *


# Function to get current memory usage in MB
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_bytes = process.memory_info().rss  # Memory usage in bytes    
    return memory_bytes

# Function to log memory usage and time
def log_memory_usage(step_name, start_memory, start_time, log_file):
    current_memory = get_memory_usage()
    current_time = time.time()
    memory_used = current_memory - start_memory
    time_elapsed = current_time - start_time
    with open(log_file, "a") as f:
        f.write(f"Step: {step_name}\n")
        f.write(f"Memory used: {(memory_used)/ (1024 * 1024):.2f} MB ({memory_used} B)\n")
        f.write(f"Time elapsed: {time_elapsed:.6f} seconds\n")
        f.write("\n")
    return current_memory, current_time

# Dubins4D TTR example 
X_RANGE = [-8.0, +8.0]
Y_RANGE = [-8.0, +8.0]
SPEED_RANGE = [-0.2, 2.2]
THETA_RANGE = [0.0, 2*np.pi]
PTS_EACH_DIM = np.array([161, 161, 25, 35])
TARGET_POSITION = [0.0, 0.0]
TARGET_RADIUS = 2.0
TARGET_SPEED = [-0.2, +0.2]
TARGET_THETA = [-0.2, +0.2]
ACCL_RANGE = [-1.0, +1.0]
OMEGA_RANGE = [-1.5, +1.5]

# Get the current Python file's name (without extension) for the log file
curret_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(curret_file_path)
current_file_name = os.path.splitext(os.path.basename(curret_file_path))[0]
log_file_name = os.path.join(current_file_dir, f"{current_file_name}_performance_log.txt")

# Initialize logging
start_time = time.time()
start_memory = get_memory_usage()

# Log initial state
with open(log_file_name, "w") as f:
    # Log variable ranges
    f.write("Variable Ranges:\n")
    f.write(f"X_RANGE: {X_RANGE}\n")
    f.write(f"Y_RANGE: {Y_RANGE}\n")
    f.write(f"SPEED_RANGE: {SPEED_RANGE}\n")
    f.write(f"THETA_RANGE: {THETA_RANGE}\n")
    f.write(f"PTS_EACH_DIM: {PTS_EACH_DIM}\n")
    f.write(f"TARGET_POSITION: {TARGET_POSITION}\n")
    f.write(f"TARGET_RADIUS: {TARGET_RADIUS}\n")
    f.write(f"TARGET_SPEED: {TARGET_SPEED}\n")
    f.write(f"TARGET_THETA: {TARGET_THETA}\n")
    f.write(f"ACCL_RANGE: {ACCL_RANGE}\n")
    f.write(f"OMEGA_RANGE: {OMEGA_RANGE}\n")
    f.write("\n")
    f.write("Initial memory usage and time:\n")
    f.write(f"Memory used: {(start_memory)/ (1024 * 1024):.2f} MB ({start_memory} B)\n")
    f.write(f"Time: {start_time:.6f} seconds\n")
    f.write("\n")

## dynamics
dubin4d = DubinsCar4D(x=[0, 0, 0, 0],
                      uMin=[ACCL_RANGE[0], OMEGA_RANGE[0]],
                      uMax=[ACCL_RANGE[1],  OMEGA_RANGE[1]],
                      dMin=[0.0, 0.0],
                      dMax=[0.0, 0.0],
                      uMode="min",
                      dMode="max",
                      )
start_memory, start_time = log_memory_usage("After creating dynamics", start_memory, start_time, log_file_name)

## grids
grid = Grid(minBounds=np.array([X_RANGE[0], Y_RANGE[0], SPEED_RANGE[0], THETA_RANGE[0]]), 
            maxBounds=np.array([X_RANGE[1], Y_RANGE[1], SPEED_RANGE[1], THETA_RANGE[1]]),
            dims=4,
            pts_each_dim=PTS_EACH_DIM,
            periodicDims=[3])
start_memory, start_time = log_memory_usage("After creating grid", start_memory, start_time, log_file_name)

## target set
target_xy = CylinderShape(grid=grid,
                        ignore_dims=[2, 3],
                        center=[TARGET_POSITION[0], TARGET_POSITION[1]],
                        radius=TARGET_RADIUS)   # x and y: Cylinder
target_else = ShapeRectangle(grid=grid,
                             target_min=[X_RANGE[0], Y_RANGE[0], TARGET_SPEED[0], TARGET_THETA[0]],
                             target_max=[X_RANGE[1], Y_RANGE[1], TARGET_SPEED[1], TARGET_THETA[1]])
target = Intersection(target_xy, target_else)
start_memory, start_time = log_memory_usage("After creating target set", start_memory, start_time, log_file_name)

## avoid set
avoid_lower = ShapeRectangle(grid=grid,
                       target_min=[X_RANGE[0], Y_RANGE[0], SPEED_RANGE[0], THETA_RANGE[0]],
                       target_max=[X_RANGE[1], Y_RANGE[1], SPEED_RANGE[0]+0.15, THETA_RANGE[1]])

avoid_upper = ShapeRectangle(grid=grid,
                       target_min=[X_RANGE[0], Y_RANGE[0], SPEED_RANGE[1]-0.15, THETA_RANGE[0]],
                       target_max=[X_RANGE[1], Y_RANGE[1], SPEED_RANGE[1], THETA_RANGE[1]])

avoid = Union(avoid_lower, avoid_upper)
start_memory, start_time = log_memory_usage("After creating avoid set", start_memory, start_time, log_file_name)

## solve
ttr_result = TTRSolver(dynamics_obj=dubin4d,
                       grid=grid,
                       multiple_value=[target_xy, avoid],
                       epsilon=0.001,
                       plot_option=PlotOptions(
                           do_plot=False,
                           plot_type="value",
                           plotDims=[0,1],
                           slicesCut=[5,5],
                       ))
start_memory, start_time = log_memory_usage("After solving TTR", start_memory, start_time, log_file_name)

# Final memory and time logging
end_time = time.time()
end_memory = get_memory_usage()
compute_time = end_time - start_time
memory_usage = end_memory - start_memory

with open(log_file_name, "a") as f:
    f.write("Final results:\n")
    f.write(f"Total compute time: {compute_time:.6f} seconds\n")
    f.write(f"Total memory usage: {memory_usage:.2f} MB\n")

print(f"Compute time: {compute_time:.6f} seconds")
print(f"Memory usage: {memory_usage:.2f} MB")