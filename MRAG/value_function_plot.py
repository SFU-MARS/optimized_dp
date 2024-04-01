import matplotlib.pyplot as plt
import numpy as np
from odp.Plots.plotting_utilities import *
from odp.Grid import Grid
from utilities import lo2slice1v1, lo2slice2v1, lo2slice1v0

# load reach-avoid value functions

# value2v1 = np.load('MRAG_Results/2v1AttackDefend_speed15.npy')
# print(f'The shape of the 2v1 value function is {value2v1.shape} \n')

value2v1 = np.load('MRAG/1v2AttackDefend_speed15.npy')
print(f'The shape of the 1v2 value function is {value2v1.shape} \n')

# value1v2 = np.load('MRAG/1v2AttackDefend_speed15.npy') 
# print(f'The shape of the 1v2 value function is {value1v2.shape} \n')


