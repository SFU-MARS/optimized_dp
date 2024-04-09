import heterocl as hcl

def min(a, *args):
    result = hcl.scalar(a, "min")
    for a in args:
        with hcl.if_(a < result.v):
            result.v = a
    return result.v

def max(a, *args):
    result = hcl.scalar(a, "max")
    for a in args:
        with hcl.if_(result.v < a):
            result.v = a
    return result.v

def abs(x):
    result = hcl.scalar(0, "abs")
    with hcl.if_(x > 0):
        result.v = x
    with hcl.else_():
        result.v = -x
    return result.v

def sign(x):
    result = hcl.scalar(0, "sign")
    with hcl.if_(x == 0):
        result.v = 0
    with hcl.if_(x > 0):
        result.v = 1
    with hcl.if_(x < 0):
        result.v = -1
    return result.v

def red_sum(x):
    # rax = hcl.reduce_axis(0, x.shape[0])
    # y = hcl.compute((1,), lambda _: hcl.sum(x[rax], axis=rax))
    y = hcl.scalar(0)
    def body(i):
        y.v += x[i]
    hcl.mutate(x.shape, body)
    return y.v

def red_max(x):
    rax = hcl.reduce_axis(0, x.shape[0])
    y = hcl.compute((1,), lambda _: hcl.max(x[rax], axis=rax))
    return y.v

def dot(x1, x2):
    assert x1.shape == x2.shape, 'Different shapes'
    y = hcl.compute(x1.shape, lambda i: x1[i] * x2[i])
    return red_sum(y)
    # rax = hcl.reduce_axis(0, y.shape[0])
    # d = hcl.compute((1,), lambda _: hcl.sum(y[rax], axis=rax))
    # return d.v

'''
def _freduce_sum(inp, out):
    pass

def sum(a, axis=None, out=None):
    """Sum along tensor axes. Similar API to `numpy.sum()`."""

    all_axes = list(range(len(a.shape)))

    if axis is None:
        axis = all_axes
    elif isinstance(axis, int):
        axis = [axis]
    elif isinstance(axis, tuple):
        axis = list(axis)
    elif not isinstance(axis, list):
        raise TypeError('Invalid type for axis argument')

    # *_axes = the index that identify an axis, e.g. 1 for the col axis of a matrix
    # *_idxs = the indices that identify an element in a tensor
    # out_shape = shape of resulting tensor
    # raxes = hcl objects for reduction axes

    red_axes = list(axis)
    raxes = [hcl.reduce_axis(0, a.shape[i], name=f'sum_rax{i}') for i in red_axes]

    out_axes = tuple(filter(lambda i: i not in axis, all_axes))
    out_shape = tuple(map(lambda i: a.shape[i], out_axes)) if out_axes else (1,)

    def rebuild_index(out_idxs):
        """Rebuild the indices again (for input tensor)."""
        idxs = tuple()
        for ax in all_axes: 
            ## There should only be one possible concatenation for each iteration, check by assertion below
            # if ax is an out axis
            idxs += tuple(out_idxs[i] for i, out_ax in enumerate(out_axes) if ax == out_ax)
            # if ax is an reduce axis
            idxs += tuple(raxes[i] for i, red_ax in enumerate(red_axes) if ax == red_ax)
        # This fails if out_axes and red_axes are not disjunct
        assert len(idxs) == len(all_axes), 'Combined index from red_axes and out_axes (should be unreachable)'
        return idxs

    out = hcl.compute(out_shape, lambda *_: 0)
    reducer = hcl.reducer(out, lambda x, y: x + y, name='sum_reducer')

    return hcl.compute(out_shape, lambda *idxs: reducer(a[*rebuild_index(idxs)], axis=raxes), name='sum_result')
'''
