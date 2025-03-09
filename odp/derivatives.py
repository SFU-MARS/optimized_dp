import heterocl as hcl
import odp.hcl_math as hcl_math

def FirstOrderENO(left, right, axis, vf, grid, *idxs):
    """1st order spatial derivative."""

    axis_len = vf.shape[axis]
    axis_step = grid.dx[axis]

    if axis in grid.periodic_dims:
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

def SecondOrderENO(left, right, axis, vf, grid, *idxs):
    """2nd order ENO spatial derivative."""

    axis_len = vf.shape[axis]
    axis_step = grid.dx[axis]

    # We need a stencil of size 5 including the current point
    V_i_minus_1 = hcl.scalar(0, 'V_i_minus_1')
    V_i_minus_2 = hcl.scalar(0, 'V_i_minus_2')
    V_i_plus_1 = hcl.scalar(0, 'V_i_plus_1')
    V_i_plus_2 = hcl.scalar(0, 'V_i_plus_2')


    if axis in grid.periodic_dims:
        with hcl.if_(idxs[axis] == 0):
            ## if n == 0 then...
            ## left deriv := (vf[n] - vf[N]) / dx
            ## right deriv := (vf[n+1] - vf[n]) / dx

            # pick last element of axis
            ix = list(idxs)
            ix[axis] = axis_len-1
            V_i_minus_1.v = vf[tuple(ix)]

            ix[axis] = axis_len - 2
            V_i_minus_2.v = vf[tuple(ix)]

            # Pick element to the right
            ix[axis] = 1
            V_i_plus_1.v = vf[tuple(ix)]

            ix[axis] = 2
            V_i_plus_2.v = vf[tuple(ix)]

        with hcl.elif_(idxs[axis] == 1):
            ## if n == 0 then...
            ## left deriv := (vf[n] - vf[N]) / dx
            ## right deriv := (vf[n+1] - vf[n]) / dx

            # pick last element of axis
            ix = list(idxs)
            ix[axis] = 0
            V_i_minus_1.v = vf[tuple(ix)]

            ix[axis] = axis_len - 1
            V_i_minus_2.v = vf[tuple(ix)]

            # Pick element to the right
            ix[axis] = 2
            V_i_plus_1.v = vf[tuple(ix)]

            ix[axis] = 3
            V_i_plus_2.v = vf[tuple(ix)]

        with hcl.elif_(idxs[axis] == axis_len-1):
            ## if n == N then...
            ## left deriv := (vf[n] - vf[n-1]) / dx
            ## right deriv := (vf[0] - vf[n]) / dx

            ix = list(idxs)
            ix[axis] = axis_len - 2
            V_i_minus_1.v = vf[tuple(ix)]

            ix[axis] = axis_len - 3
            V_i_minus_2.v = vf[tuple(ix)]

            # Pick element to the right
            ix[axis] = 0
            V_i_plus_1.v = vf[tuple(ix)]

            ix[axis] = 1
            V_i_plus_2.v = vf[tuple(ix)]

        with hcl.elif_(idxs[axis] == axis_len-2):
            ## if n == N then...
            ## left deriv := (vf[n] - vf[n-1]) / dx
            ## right deriv := (vf[0] - vf[n]) / dx

            ix = list(idxs)
            ix[axis] = axis_len - 3
            V_i_minus_1.v = vf[tuple(ix)]

            ix[axis] = axis_len - 4
            V_i_minus_2.v = vf[tuple(ix)]

            # Pick element to the right
            ix[axis] = axis_len - 1
            V_i_plus_1.v = vf[tuple(ix)]

            ix[axis] = 0
            V_i_plus_2.v = vf[tuple(ix)]

        with hcl.else_():

            ix = list(idxs)
            ix[axis] = axis_len - 2
            V_i_minus_1.v = vf[tuple(ix)]

            ix[axis] = axis_len - 1
            V_i_minus_2.v = vf[tuple(ix)]

            # Pick element to the right
            ix[axis] = axis_len + 1
            V_i_plus_1.v = vf[tuple(ix)]

            ix[axis] = axis_len + 2
            V_i_plus_2.v = vf[tuple(ix)]

    else:
        # left boundary  = vf[n] + sign(vf[n]) * |vf[n+1] - vf[n]|
        # right boundary = vf[n] + sign(vf[n]) * |vf[n] - vf[n-1]|
        with hcl.if_(idxs[axis] == 0):
            ## if n == 0 then...
            ## left deriv := (vf[n] - LB) / dx
            ## right deriv := (vf[n+1] - vf[n]) / dx

            # Need to extrapolate
            ix = list(idxs)
            ix[axis] += 1
            ix = tuple(ix)
            V_i_minus_1.v = vf[idxs] + hcl_math.sign(vf[idxs]) * hcl_math.abs(vf[ix] - vf[idxs])
            V_i_minus_2.v = vf[idxs] + 2 * hcl_math.sign(vf[idxs]) * hcl_math.abs(vf[ix] - vf[idxs])

            # Pick element to the right
            ix = list(idxs)
            ix[axis] = 1
            V_i_plus_1.v = vf[tuple(ix)]

            ix[axis] = 2
            V_i_plus_2.v = vf[tuple(ix)]

        with hcl.elif_(idxs[axis] == 1):
            ## if n == N then...
            ## left deriv := (vf[n] - vf[n-1]) / dx
            ## right deriv := (RB - vf[n]) / dx

            ix = list(idxs)
            ix[axis] = 0
            V_i_minus_1.v = vf[tuple(ix)]
            # Need to extrapolate
            V_i_minus_2.v = vf[tuple(ix)] + hcl_math.sign(vf[tuple(ix)]) * hcl_math.abs(vf[tuple(ix)] - vf[idxs])

            # Pick element to the right
            ix = list(idxs)
            ix[axis] = 2
            V_i_plus_1.v = vf[tuple(ix)]

            ix[axis] = 3
            V_i_plus_2.v = vf[tuple(ix)]

        with hcl.elif_(idxs[axis] == axis_len - 1):
            ## if n == N then...
            ## left deriv := (vf[n] - vf[n-1]) / dx
            ## right deriv := (RB - vf[n]) / dx

            ix = list(idxs)
            ix[axis] = axis_len - 2
            V_i_minus_1.v = vf[tuple(ix)]

            ix[axis] = axis_len - 3
            V_i_minus_2.v = vf[tuple(ix)]

            # Pick element to the right
            V_i_plus_1.v = vf[idxs] + hcl_math.sign(vf[idxs]) * hcl_math.abs(vf[idxs] - V_i_minus_1)
            # Need to extrapolate
            V_i_plus_2.v = vf[idxs] + 2 * hcl_math.sign(vf[idxs]) * hcl_math.abs(vf[idxs] - V_i_minus_1)

        with hcl.elif_(idxs[axis] == axis_len - 2):
            ## if n == N then...
            ## left deriv := (vf[n] - vf[n-1]) / dx
            ## right deriv := (RB - vf[n]) / dx

            ix = list(idxs)
            ix[axis] = axis_len - 3
            V_i_minus_1.v = vf[tuple(ix)]

            ix[axis] = axis_len - 4
            V_i_minus_2.v = vf[tuple(ix)]

            # Pick element to the right
            ix = list(idxs)
            ix[axis] = axis_len - 1
            V_i_plus_1.v = vf[tuple(ix)]
            V_i_plus_2.v = vf[tuple(ix)] + hcl_math.sign(vf[tuple(ix)]) * hcl_math.abs(vf[tuple(ix)] - vf[idxs])

        with hcl.else_():

            ix = list(idxs)
            ix[axis] = axis_len - 2
            V_i_minus_1.v = vf[tuple(ix)]

            ix[axis] = axis_len - 1
            V_i_minus_2.v = vf[tuple(ix)]

            # Pick element to the right
            ix[axis] = axis_len + 1
            V_i_plus_1.v = vf[tuple(ix)]

            ix[axis] = axis_len + 2
            V_i_plus_2.v = vf[tuple(ix)]


    D1_minus_2_plus_half = (V_i_minus_1.v - V_i_minus_2.v) / axis_step
    D1_minus_1_plus_half = (vf[idxs] - V_i_minus_1.v) / axis_step
    D1_0_plus_half = (V_i_plus_1.v - vf[idxs]) / axis_step
    D1_plus_1_plus_half = (V_i_plus_2.v - V_i_plus_1.v) / axis_step

    D2_minus_1 = (D1_minus_1_plus_half - D1_minus_2_plus_half) / (2 * axis_step)
    D2_0 = (D1_0_plus_half - D1_minus_1_plus_half) / (2 * axis_step)
    D2_plus_1 = (D1_plus_1_plus_half - D1_0_plus_half) / (2 * axis_step)

    #
    with hcl.if_(hcl_math.abs(D2_minus_1) <= hcl_math.abs(D2_0)):
        left.v = D1_minus_1_plus_half + D2_minus_1 * axis_step
    with hcl.else_():
        left.v = D1_minus_1_plus_half + D2_0 * axis_step

    with hcl.if_(hcl_math.abs(D2_0) <= hcl_math.abs(D2_plus_1)):
        right.v = D1_0_plus_half - D2_0 * axis_step
    with hcl.else_():
        right.v = D1_0_plus_half - D2_plus_1 * axis_step

    return left.v, right.v
