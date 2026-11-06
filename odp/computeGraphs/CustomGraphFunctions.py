import heterocl as hcl
import numpy as np

# Whether the Dissipation stage must evaluate the optimal disturbance at every grid point.
# Dynamics classes whose opt_dstb does not depend on the state can set
# `dstb_state_dependent = False` so it is evaluated once per time step instead.
# Defaults to True (exact for any dynamics).
def dstb_depends_on_state(my_object):
    return getattr(my_object, "dstb_state_dependent", True)

# Custom function

def my_min(a,b):
    result = hcl.scalar(0, "result")
    with hcl.if_(a < b):
        result[0] = a
    with hcl.else_():
        result[0] = b
    return result[0]

# Input is value, output is value
def my_max(a,b):
    result = hcl.scalar(0, "result")
    with hcl.if_(a > b):
        result.v = a
    with hcl.else_():
        result.v = b
    return result.v

# Input is value, output is value
def max_4n(num1, num2, num3, num4):
    largest = hcl.scalar(0, "largest")
    largest[0] = my_max(num1, num2)
    largest[0] = my_max(largest[0], num3)
    largest[0] = my_max(largest[0], num4)

    return largest[0]

# Input is a value, output is value
def my_abs(my_x):
    abs_value = hcl.scalar(0, "abs_value", dtype=hcl.Float())
    with hcl.if_(my_x > 0):
        abs_value.v = my_x
    with hcl.else_():
        abs_value.v = -my_x
    return abs_value.v

def new_abs(my_x):
    my_value = hcl.scalar(0, "my_value", dtype=hcl.Float())
    with hcl.if_(my_x > 0):
        my_value.v = my_x
    with hcl.else_():
        my_value.v = -my_x
    return my_value.v

def my_sign(x):
    sign = hcl.scalar(0, "sign", dtype=hcl.Float())
    with hcl.if_(x == 0):
        sign[0] = 0
    with hcl.if_(x > 0):
        sign[0] = 1
    with hcl.if_(x < 0):
        sign[0] = -1
    return sign[0]

