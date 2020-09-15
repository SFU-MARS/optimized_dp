import heterocl as hcl
import numpy as np
import time
from user_definer import *
# Custom function

def my_atan(x):
    my_st_result = hcl.scalar(0, "my_st_result")
    # Pay attention to the sign
    with hcl.if_(x <= 1):
        with hcl.if_(x >= -1):
            my_st_result[0] = x - x*x*x/3 + x*x*x*x*x/5 - x*x*x*x*x*x*x/7 + x*x*x*x*x*x*x*x*x/9 \
                             - x*x*x*x*x*x*x*x*x*x*x/11 + x*x*x*x*x*x*x*x*x*x*x*x*x/13 \
                        - x*x*x*x*x*x*x*x*x*x*x*x*x*x*x/15 + x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x/17 \
            - x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x/19 + x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x/21\
            - x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x/23 + x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x/25

        with hcl.elif_(x < -1):
            my_st_result[0] = -math.pi/2 -(1/x - 1/(x*x*x*3) + 1/(x*x*x*x*x*5) - 1/(x*x*x*x*x*x*x*7) + 1/(x*x*x*x*x*x*x*x*x*9) \
                             - 1/(x*x*x*x*x*x*x*x*x*x*x*11) + 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*13) \
                        - 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*15) + 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*17) \
            - 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*19) + 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*21)\
            - 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*23) + 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*25))
    with hcl.if_(x > 1):
            my_st_result[0] = math.pi/2 -(1/x - 1/(x*x*x*3) + 1/(x*x*x*x*x*5) - 1/(x*x*x*x*x*x*x*7) + 1/(x*x*x*x*x*x*x*x*x*9) \
                             - 1/(x*x*x*x*x*x*x*x*x*x*x*11) + 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*13) \
                        - 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*15) + 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*17) \
            - 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*19) + 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*21)\
            - 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*23) + 1/(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*25))

    return my_st_result[0]

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