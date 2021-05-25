import numpy as np
import math
import os
#import MDP.Example_3D as MDP
import MDP.Example_6D as MDP
from solver import *

myProblem   = MDP.MDP_6D_example()
solveValueIteration(myProblem)

# Optionally provide a directory and filename to save results of computation
dir_path   = None
file_name  = None

