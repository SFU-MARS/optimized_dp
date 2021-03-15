import math

import PIL
import numpy as np
reso=2
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
#for spatial

# grid_x, grid_y, grid_v, grid_theta = np.mgrid[0:30:reso*600j, 0:26.05:reso*521j,-1:3:31j ,-math.pi:-math.pi:9j ]
# points=(np.mgrid[0:30:600j, 0:26.05:521j,-1:3:31j ,-math.pi:-math.pi:9j],
# obstaclemap_4D=np.load("/local-scratch/tara/project/WayPtNav-reachability/obstacle_grid_4d.npy")
# obstaclemap_2D=obstaclemap_4D[:,:,30,0]
# from scipy.interpolate import griddata
# grid_z0 = griddata(points, values, (grid_x, grid_y, grid_v, grid_theta), method='nearest')
data=np.load("/home/ttoufigh/optimized_dp/TTR_grid_biggergrid_3lookback_wDisturbance_wObstalceMap_speedlimit3reverse_5.npy")
# data=np.load("/home/ttoufigh/optimized_dp/TTR_grid_biggergrid_3lookback_wDisturbance_wObstalceMap_speedlimit3_5.npy")
vind=26#27=speed3, 10=speed 0
v_range=np.arange(-1.5,3.5, 1/6)
# print(v_range[vind,])
TTC=data[:,:,vind,:]
# TTCm=np.max(np.max(TTC))
# print(TTCm)
# import matplotlib
# n_cols = ['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180']
# fig, ax = plt.subplots(3, 3)
# grid size [600, 521, 31,9]
x1=445
x2=460
y1=240
y2=260
# xind_range=x1:x2
# yind_range=y1:y2
# xind_range=list(np.arange(420,440))
# yind_range=list(np.arange(305,325))
from PIL import Image
# TTC1=TTC[:, :,0]
# plt.matshow(TTC1)
# plt.show()
# plt.imshow(),
# figure, ax = pyplot.subplots(1)
# rect = patches.Rectangle((125,100),50,25, edgecolor='r', facecolor="none")

# TTC1=TTC[x1:x2, y1:y2,0]
# fromarray(TTC1.show()

ax.add_patch(rect)
# image =PIL.Image.fromarray(TTC1, "RGB")
# plt.imshow(image)
# plt.show()
# pos0=ax[0,1].matshow(TTC1)
# # ax[0,0].set_title('theta=-180')
# ax[0,1].set_title('theta=90')#(-180-90)%360
# ax[0,1].title.set_position([.1,1.2])
# plt.show()
# img = Image.fromarray(TTC1)
# img.show()

# pos0=ax[0,0].matshow(TTC[xind_range,yind_range,1], label='-135')
# pos0=ax[0,0].matshow(TTC[420:440,305:325,1])
# from PIL import Image
# img = Image.fromarray(pos0)
# img.show()
pos0=ax[0,0].matshow(TTC[:,:,1])
# ax[0,0].set_title('theta=-135')
ax[0,0].set_title('theta=135')
ax[0,0].title.set_position([.1,1.2])
# plt.show()

# p0, = plt.plot([1, 2, 3], label='-135')
# plt.legend(handles=[p0], bbox_to_anchor=(1.05, 1), loc='upper left')
fig.colorbar(pos0, ax=ax[0,0])
plt.show()

pos01=ax[0,1].matshow(TTC[:,:,0],label='theta=180')
fig.colorbar(pos01, ax=ax[0,1])
# ax[0,1].set_title('theta=180')
ax[0,1].set_title('theta=90')
ax[0,1].title.set_position([.1,1.2])

pos02=ax[0,2].matshow(TTC[:,:,7], label='theta=135')
fig.colorbar(pos02, ax=ax[0,2])
# ax[0,2].set_title('theta=135')
ax[0,2].set_title('theta=45')
ax[0,2].title.set_position([.1,1.2])

ax[1,0].matshow(TTC[:,:,2], label='theta=-90')
fig.colorbar(pos02, ax=ax[1,0])
# ax[1,0].set_title('theta=-90')
ax[1,0].set_title('theta=180')
ax[1,0].title.set_position([.1,1.2])
# ax[1,1].matshow(obstaclemap_2D, label='obstacle map')
ax[1,2].matshow(TTC[:,:,6], label='theta=90')
fig.colorbar(pos02, ax=ax[1,2])
# ax[1,2].set_title('theta=90')
ax[1,2].set_title('theta=0')
ax[1,2].title.set_position([.1,1.2])
ax[2,0].matshow(TTC[:,:,3], label='theta=-45')
fig.colorbar(pos02, ax=ax[2,0])
# ax[2,0].set_title('theta=-45')
ax[2,0].set_title('theta=225')
ax[2,0].title.set_position([.1,1.2])
ax[2,1].matshow(TTC[:,:,4], label='theta=0')
fig.colorbar(pos02, ax=ax[2,1])
# ax[2,1].set_title('theta=0')
ax[2,1].set_title('theta=-90')
ax[2,1].title.set_position([.1,1.2])
ax[2,2].matshow(TTC[:,:,5], label='theta=45')
fig.colorbar(pos02, ax=ax[2,2])
# ax[2,2].set_title('theta=45')
ax[2,2].set_title('theta=-45')
ax[2,2].title.set_position([.1,1.2])

plt.suptitle('v=Vhigh=2.7, dxy=0.05 , dtheta=0.15 , uw=0.4, ua=1.1', x=0.1, y=.95, horizontalalignment='left', verticalalignment='bottom', fontsize = 15)


plt.show()
# fig.tight_layout()
# ax[1,1].set_title('v=vmax=3')
