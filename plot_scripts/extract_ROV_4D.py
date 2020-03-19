import numpy as np
import scipy.io as sio

for i in range(1, 101):
    fileName = '/local-scratch/ROV_6d_TEB/V_value_{:d}'.format(i)
    fileRead = sio.loadmat(fileName)
    V = fileRead['V_array']
    V = np.amax(V, axis = (2,3))
    print("Saving file {:d}".format(i))
    sio.savemat('/local-scratch/ROV_4D_TEB/V_value_{:d}.mat'.format(i), {'V_array': V})

print('Finished converting')
