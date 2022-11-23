import numpy as np


# locations 2 slices
def loca2slices(x_location, y_location, slices=45):
    x_slice = np.round((1 + x_location) * (slices - 1) / 2)
    y_slice = np.round((1 + y_location) * (slices - 1) / 2)
    return int(x_slice), int(y_slice)


# calculate the value function of the current state
def state_value(V, x1, y1, x2, y2, slices=45):
    # (x1, y1) and (x2, y2) are locations
    x1_slice, y1_slice = loca2slices(x1, y1, slices)
    x2_slice, y2_slice = loca2slices(x2, y2, slices)
    value = V[x1_slice, y1_slice, x2_slice, y2_slice, 0]  # 0 means the final tube
    return value


# check in the current state, the attacker is captured by the defender or not
def check(V, x1, y1, x2, y2, slices=45):
    # (x1, y1) and (x2, y2) are locations
    flag = state_value(V, x1, y1, x2, y2, slices=slices)
    if flag >= 0:
        return 1
    else:
        return 0



