import numpy as np
import math
import heterocl as hcl
import os

from odp.solver import *
from odp.Grid import Grid

# Use symmetric bounds with an odd number of points (201).
# This guarantees that the origin (0.0, 0.0) is an exact grid node.
bounds = np.array([[-2., 2.], [-2., 2.]])
ptsEachDim = np.array([601, 601])

g = Grid(minBounds=bounds[:, 0],
         maxBounds=bounds[:, 1],
         dims=2,
         pts_each_dim=ptsEachDim)

# Action space (acceleration limits)
actions = np.linspace(-6., 6., 41)

# gamma = 1.0 is required to match the infinite-horizon LQR cost function
gamma = np.array([1.])
epsilon = np.array([1.117e-5])
maxIters = np.array([2000]) # May take a few more iterations to fully propagate with gamma=1.0

class double_integrator_lqr:
    def __init__(self):
        self.dt = 0.05
        self.max_x1 = bounds[0, 1]
        self.max_x2 = bounds[1, 1]
        self.maxTransitions = 1 

    def transition(self, sVals, iVals, u):
        trans_matrix = hcl.compute((self.maxTransitions, (1 + 2)), lambda *x: 0, "trans_matrix")
        
        # Variable declaration
        x1 = hcl.scalar(0, "x1")
        x2 = hcl.scalar(0, "x2")
        new_x1 = hcl.scalar(0, "new_x1")
        new_x2 = hcl.scalar(0, "new_x2")

        x1[0] = sVals[0]
        x2[0] = sVals[1]

        # Absorbing state logic: if exactly at the origin, you stay at the origin
        with hcl.if_(x1[0] * x1[0] + x2[0] * x2[0] == 0.0):
            new_x1[0] = 0.0
            new_x2[0] = 0.0
        with hcl.else_():
            # Exact Zero-Order Hold (ZOH) Discrete Dynamics
            new_x1[0] = x1[0] + x2[0] * self.dt #+ 0.5 * u * self.dt * self.dt
            new_x2[0] = x2[0] + u * self.dt
        
        # Bound the new states to stay within grid limits
        with hcl.if_(new_x1[0] > self.max_x1):
            new_x1[0] = self.max_x1
        with hcl.if_(new_x1[0] < -self.max_x1):
            new_x1[0] = -self.max_x1
            
        with hcl.if_(new_x2[0] > self.max_x2):
            new_x2[0] = self.max_x2
        with hcl.if_(new_x2[0] < -self.max_x2):
            new_x2[0] = -self.max_x2

        # Probability of 1 for deterministic transition 
        trans_matrix[0, 0] = 1.0
        trans_matrix[0, 1] = new_x1[0]
        trans_matrix[0, 2] = new_x2[0]

        return trans_matrix

    def reward(self, sVals, iVals, u):
        rwd = hcl.scalar(0, "rwd")
        x1 = hcl.scalar(0, "x1")
        x2 = hcl.scalar(0, "x2")
        
        x1[0] = sVals[0]
        x2[0] = sVals[1]

        # Goal state has zero cost. Everywhere else accumulates negative reward.
        with hcl.if_(x1[0] * x1[0] + x2[0] * x2[0] == 0.0):
            rwd[0] = 0.0
        with hcl.else_():
            # LQR Cost Formulation: -(x^T Q x + u^T R u) * dt
            # Assuming Q = I (Identity), R = 1
            rwd[0] = -(x1[0] * x1[0] + x2[0] * x2[0] + u * u) * self.dt
            
        return rwd[0]

lqr_system = double_integrator_lqr()
result = solveValueIteration(lqr_system,
                             grid=g, action_space=actions,
                             gamma=gamma, epsilon=epsilon,
                             maxIters=maxIters)

np.save('hcl_lqr_res.npy', result)
import matplotlib.pyplot as plt

V_numeric = np.load('hcl_lqr_res.npy')

# 2. Recreate the grid coordinates (201 points between -2 and 2)
# Using 'ij' indexing to match how optimizeddp handles multi-dimensional arrays
x1 = np.linspace(bounds[0, 0], bounds[0, 1], ptsEachDim[0])
x2 = np.linspace(bounds[1, 0], bounds[1, 1], ptsEachDim[1])
X1, X2 = np.meshgrid(x1, x2, indexing='ij')

# 3. Compute the Exact Analytical LQR Solution
# Note: Negated because our MDP reward formulation maximized negative cost
V_exact = -(np.sqrt(3) * X1**2 + 2 * X1 * X2 + np.sqrt(3) * X2**2)

# 4. Calculate the Absolute Error
error = np.abs(V_numeric - V_exact)

# 5. Plot the Results
fig = plt.figure(figsize=(18, 5))

# Plot A: The Numeric Value Function from optimizeddp
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X1, X2, V_numeric, cmap='viridis', edgecolor='none')
ax1.set_title('Numeric Value Function (optimizeddp)')
ax1.set_xlabel('Position ($x_1$)')
ax1.set_ylabel('Velocity ($x_2$)')
ax1.set_zlabel('Value')

# Plot B: The Ground Truth Analytical Value Function
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X1, X2, V_exact, cmap='viridis', edgecolor='none')
ax2.set_title('Exact LQR Value Function')
ax2.set_xlabel('Position ($x_1$)')
ax2.set_ylabel('Velocity ($x_2$)')
ax2.set_zlabel('Value')

# Plot C: Error Heatmap
ax3 = fig.add_subplot(133)
# Use a contour plot to show where the error is highest (typically at grid boundaries)
contour = ax3.contourf(X1, X2, error, levels=50, cmap='magma')
cbar = fig.colorbar(contour, ax=ax3)
cbar.set_label('Absolute Error')
ax3.set_title(r'Absolute Error $|V_{num} - V_{exact}|$')
ax3.set_xlabel('Position ($x_1$)')
ax3.set_ylabel('Velocity ($x_2$)')

plt.tight_layout()

# Save the plots instead of showing them
plt.savefig('lqr_convergence_plots.pdf', dpi=300, bbox_inches='tight') # Best for LaTeX papers
plt.savefig('lqr_convergence_plots.png', dpi=300, bbox_inches='tight') # Easy to view in VS Code
plt.close(fig) # Free up memory

# 6. Calculate Error ONLY on the Interior (Avoid boundary pollution)
# Adaptively find the indices closest to -1.0 and 1.0
x1 = g.grid_points[0]
idx_min = np.argmin(np.abs(x1 - (-1.0)))
idx_max = np.argmin(np.abs(x1 - 1.0))

# Slice the error array to only look at the safe interior center
# Add +1 to idx_max to ensure the upper bound is inclusive in the slice
interior_error = error[idx_min:idx_max+1, idx_min:idx_max+1]

# Calculate the new L-infinity norm (Max error)
L_inf_error_interior = np.max(interior_error)
L_1_error_interior = np.mean(interior_error)

print(f"Full Grid L-infinity Error: {np.max(error):.4f} (Polluted by control bounds)")
print(f"Interior [-1, 1] L-infinity Error: {L_inf_error_interior:.4f}")
print(f"Interior [-1, 1] L-1 (Mean) Error: {L_1_error_interior:.4f}")
print("Saved plots to lqr_convergence_plots.pdf and lqr_convergence_plots.png")
# --- SIDE-BY-SIDE CONTOUR PLOT COMPARISON (Restricted to [-1, 1]) ---
# Slice the coordinate grids and value functions to the [-1, 1] region
X1_sub = X1[idx_min:idx_max+1, idx_min:idx_max+1]
X2_sub = X2[idx_min:idx_max+1, idx_min:idx_max+1]
V_numeric_sub = V_numeric[idx_min:idx_max+1, idx_min:idx_max+1]
V_exact_sub = V_exact[idx_min:idx_max+1, idx_min:idx_max+1]

# 1. Establish a shared color scale for a fair visual comparison
# Find the absolute min and max across BOTH surfaces in the restricted region
vmin = min(np.min(V_numeric_sub), np.min(V_exact_sub))
vmax = max(np.max(V_numeric_sub), np.max(V_exact_sub))
levels = np.linspace(vmin, vmax, 40) # 40 contour levels for smoothness

# 2. Set up the figure with 1 row and 2 columns
fig_contour, (ax_num, ax_exact) = plt.subplots(1, 2, figsize=(14, 6))

# Plot A: Numeric Contour (optimizeddp) on the restricted grid
contour_num = ax_num.contourf(X1_sub, X2_sub, V_numeric_sub, levels=levels, cmap='viridis')
ax_num.set_title('Numeric Value Function (optimizeddp)')
ax_num.set_xlabel('Position ($x_1$)')
ax_num.set_ylabel('Velocity ($x_2$)')
ax_num.grid(True, linestyle='--', alpha=0.5) # Optional: adds a faint grid for readability

# Plot B: Exact Analytical Contour on the restricted grid
contour_exact = ax_exact.contourf(X1_sub, X2_sub, V_exact_sub, levels=levels, cmap='viridis')
ax_exact.set_title('Exact LQR Value Function')
ax_exact.set_xlabel('Position ($x_1$)')
ax_exact.set_ylabel('Velocity ($x_2$)')
ax_exact.grid(True, linestyle='--', alpha=0.5)

# 3. Add a single, unified colorbar to the right of the plots
cbar = fig_contour.colorbar(contour_num, ax=[ax_num, ax_exact], fraction=0.05, pad=0.04)
cbar.set_label('Value Function $V(x)$')

plt.tight_layout()

# 4. Save the figure
plt.savefig('lqr_contour_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('lqr_contour_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig_contour)