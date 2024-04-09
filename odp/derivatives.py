import heterocl as hcl
from . import math as hcl_math

def spatial_derivative(left, right, axis, vf, grid, *idxs):
    """1st order spatial derivative."""

    axis_len = vf.shape[axis]
    axis_step = grid.dx[axis]

    if grid.periodic_dims[axis]:
        with hcl.if_(idxs[axis] == 0):
            ## if n == 0 then...
            ## left deriv := (vf[n] - vf[N]) / dx
            ## right deriv := (vf[n+1] - vf[n]) / dx

            # pick last element of axis
            ix = list(idxs)
            ix[axis] = axis_len-1
            ix = tuple(ix)
            left.v = vf[ix]
            left.v = (vf[idxs] - left.v) / axis_step

            # pick element to the right
            ix = list(idxs)
            ix[axis] += 1
            ix = tuple(ix)
            right.v = (vf[ix] - vf[idxs]) / axis_step

        with hcl.elif_(idxs[axis] == axis_len-1):
            ## if n == N then...
            ## left deriv := (vf[n] - vf[n-1]) / dx
            ## right deriv := (vf[0] - vf[n]) / dx

            # pick element to the left
            ix = list(idxs)
            ix[axis] -= 1
            ix = tuple(ix)
            left.v = (vf[idxs] - vf[ix]) / axis_step

            # pick first element of axis
            ix = list(idxs)
            ix[axis] = 0
            ix = tuple(ix)
            right.v = vf[ix] # right boundary
            right.v = (right.v - vf[idxs]) / axis_step

        with hcl.else_():

            # pick element to the left
            ix = list(idxs)
            ix[axis] -= 1
            ix = tuple(ix)
            left.v = (vf[idxs] - vf[ix]) / axis_step

            # pick element to the right
            ix = list(idxs)
            ix[axis] += 1
            ix = tuple(ix)
            right.v = (vf[ix] - vf[idxs]) / axis_step

    else:

        # left boundary  = vf[n] + sign(vf[n]) * |vf[n+1] - vf[n]|
        # right boundary = vf[n] + sign(vf[n]) * |vf[n] - vf[n-1]|

        with hcl.if_(idxs[axis] == 0):
            ## if n == 0 then... 
            ## left deriv := (vf[n] - LB) / dx
            ## right deriv := (vf[n+1] - vf[n]) / dx

            # left boundary
            ix = list(idxs)
            ix[axis] += 1
            ix = tuple(ix)
            left.v = vf[idxs] + hcl_math.sign(vf[idxs]) * hcl_math.abs(vf[ix] - vf[idxs])
            left.v = (vf[idxs] - left.v) / axis_step

            # element to the right
            ix = list(idxs) 
            ix[axis] += 1
            ix = tuple(ix)
            right.v = (vf[ix] - vf[idxs]) / axis_step

        with hcl.elif_(idxs[axis] == axis_len-1):
            ## if n == N then...
            ## left deriv := (vf[n] - vf[n-1]) / dx
            ## right deriv := (RB - vf[n]) / dx

            # element to the left 
            ix = list(idxs)
            ix[axis] -= 1
            ix = tuple(ix)
            left.v = (vf[idxs] - vf[ix]) / axis_step

            # right boundary
            ix = list(idxs)
            ix[axis] -= 1
            ix = tuple(ix)
            right.v = vf[idxs] + hcl_math.sign(vf[idxs]) * hcl_math.abs(vf[idxs] - vf[ix])
            right.v = (right.v - vf[idxs]) / axis_step

        with hcl.else_():

            # element to the left
            ix = list(idxs)
            ix[axis] -= 1
            ix = tuple(ix)
            left.v = (vf[idxs] - vf[ix]) / axis_step

            # element to the right
            ix = list(idxs)
            ix[axis] += 1
            ix = tuple(ix)
            right.v = (vf[ix] - vf[idxs]) / axis_step

    return left.v, right.v

