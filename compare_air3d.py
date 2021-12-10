
# %% 
import numpy as np
import math
from Grid.GridProcessing import Grid
from dynamics.Air3D import Air3D
import scipy.io as spio

g = Grid(np.array([-1.0, -1.0, -math.pi]), np.array([1.0, 1.0, math.pi]), 3, np.array([101, 101, 101]), [2])

V_opt_dp = np.load('V_air3D_0.25.npy')

THETA = 1.5863
# %% 
true_BRT_path = './analytical_BRT_air3D.mat'
helper_oc = spio.loadmat(true_BRT_path)

# %%
theta_values = helper_oc['gmat'][0, 0, :, 2]
print(theta_values)
theta_idx = np.argmin(abs(theta_values - THETA))

# %%
assert theta_idx == np.argmin(abs(g.grid_points[2] - THETA))
# %%
valfunc_true = helper_oc['data'][:, :, theta_idx, 10]
brs_actual = (valfunc_true <= 0.001) * 1.

# %%

# %%

import matplotlib.pyplot as plt
plt.imshow(brs_actual.T, cmap='seismic', alpha=0.5, origin='lower', vmin=-1., vmax=1., extent=(-1., 1., -1., 1.), interpolation='bilinear')
# %%
valfunc_true_opt_dp = V_opt_dp[:, :, theta_idx]
brs_actual_opt_dp = (valfunc_true_opt_dp <= 0.001) * 1.
plt.imshow(brs_actual_opt_dp.T, cmap='seismic', alpha=0.5, origin='lower', vmin=-1., vmax=1., extent=(-1., 1., -1., 1.), interpolation='bilinear')

# %%
from Plots.plotting_utilities import *
from plot_options import *
po = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0,1,2], slicesCut=[])
# plot_isosurface(g, V_opt_dp, po)

# %%
plot_isosurface(g, helper_oc['data'][:, :, :, -1], po)


# %%
