import heterocl as hcl
import numpy as np
import time
from user_definer import *
# Custom function

def my_arctan(a):
    result = hcl.scalar(0, "result", dtype=hcl.Float())
    result.v = np.arctan(a)
    return result.v

# Input is a value, output is value
def my_abs(my_x):
    abs_value = hcl.scalar(0, "abs_value", dtype=hcl.Float())
    with hcl.if_(my_x > 0):
        abs_value.v = my_x
    with hcl.else_():
        abs_value.v = -my_x
    return abs_value.v

def my_min(a,b):
    result = hcl.scalar(0, "result")
    with hcl.if_(a < b):
        result[0] = a
    with hcl.else_():
        result[0] = b
    return result[0]