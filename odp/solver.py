import heterocl as hcl
import numpy as np
import time

from . import math as hcl_math
from .derivatives import spatial_derivative
from .grid import Grid
from .shapes import *

class Solver: 

    debug = False
    interactive = False
    
    accuracy = 'low'
    
    _executable = None

    def __init__(self, grid, model, *, 
                 interactive=True,
                 debug=False,
                 accuracy='low',
                 dtype=hcl.Float()):

        # Solver options
        self.interactive = interactive
        self.debug = debug

        if self.interactive:
            print("== Welcome to optimized_dp ==")

        # Initialize the HCL environment
        hcl.init(hcl.Float(32))

        self.accuracy = accuracy
        assert self.accuracy == 'low', 'This modification to odp only supports low accuracy'

        self.dtype = dtype

        self.grid = grid
        self.model = model

        self.state_shape = (self.model.state_dims,)
        self.ctrl_shape = (self.model.ctrl_dims,)
        self.dstb_shape = (self.model.dstb_dims,)

        self.build()
        
    def __call__(self, *args):
        """
        Run the solver.
        
        This method is intended to be overloaded.
        """

        # Run the executable
        self._executable(*args)
        
    def build(self):

        if self.interactive:
            print('Building...')

        vf = hcl.placeholder(self.grid.shape, name="vf", dtype=self.dtype)
        t = hcl.placeholder((2,), name="t", dtype=self.dtype)
        xs = [hcl.placeholder((axlen,), dtype=self.dtype, name=f'x_{i}')
              for i, axlen in enumerate(self.grid.shape)]
        args = [vf, t, *xs]

        if self.debug:
            h = hcl.placeholder(self.grid.shape, name='h', dtype=self.dtype)
            args = [h] + args

        # lambda is necessary so that hcl can modify properties of the function object
        program = lambda *args: self.entrypoint(*args)
        self._sched = hcl.create_schedule(args, program)

        if not self.debug:

            # Accessing the hamiltonian and dissipation stage
            stage_hamiltonian = program.Hamiltonian
            stage_dissipation = program.Dissipation

            # Thread parallelize hamiltonian and dissipation computation
            self._sched[stage_hamiltonian].parallel(stage_hamiltonian.axis[0])
            self._sched[stage_dissipation].parallel(stage_dissipation.axis[0])

        self._executable = hcl.build(self._sched)

        if self.interactive:
            print(f'> {type(self).__name__} built!\n')

    def entrypoint(self, *args):

        if self.debug:
            h_dbg, vf, t, *xs = args
        else:
            vf, t, *xs = args

        # Initialize intermediate tensors
        dv_diff = hcl.compute(self.grid.shape + self.state_shape, lambda *_: 0, name='dv_diff')
        dv_max = hcl.compute(self.state_shape, lambda _: -1e9, name='dv_max')
        dv_min = hcl.compute(self.state_shape, lambda _: +1e9, name='dv_min')
        max_alpha = hcl.compute(self.state_shape, lambda _: -1e9, name='max_alpha')

        # Initialize the Hamiltonian tensor
        h = hcl.compute(self.grid.shape, lambda *idxs: vf[idxs], name='h')

        # Compute Hamiltonian term, max and min derivative
        self.hamiltonian_stage(h, vf, t, xs,
                               dv_min=dv_min, dv_max=dv_max, dv_diff=dv_diff)
        
        if self.debug:
            hcl.update(h_dbg, lambda *idxs: h[idxs])

        # Compute artificial dissipation
        self.dissipation_stage(h, t, xs,
                               dv_min=dv_min, dv_max=dv_max, dv_diff=dv_diff,
                               max_alpha=max_alpha)

        # Compute integration time step
        delta_t = hcl.compute((1,), lambda _: self.time_step(t, max_alpha=max_alpha))

        # First order Runge-Kutta (RK) integrator
        hcl.update(vf, lambda *idxs: vf[idxs] + h[idxs] * delta_t.v)


    def hamiltonian_stage(self, h, vf, t, xs, *,
                          dv_min, dv_max, dv_diff):
        """Calculate Hamiltonian for every grid point in V_init"""

        def body(*idxs):

            u = hcl.compute(self.ctrl_shape, lambda _: 0, name='u')
            d = hcl.compute(self.dstb_shape, lambda _: 0, name='d')
            x = hcl.compute(self.state_shape, lambda _: 0, name='x')
            dx = hcl.compute(self.state_shape, lambda _: 0, name='dx')
            dv = hcl.compute(self.state_shape, lambda _: 0, name='dv_avg')

            # x_n = X_{n,i} where 
            #   x = `x` The state tensor,
            #   X = `xs` The list of state space arrays,
            #   n = Current state dimension (in updating x),
            #   i = `idxs[n]` Index of current grid point in dimension `n`,
            for n, i in enumerate(idxs):
                x[n] = xs[n][i]

            for axis in range(self.grid.ndims):

                left = hcl.scalar(0, 'left')
                right = hcl.scalar(0, 'right')

                # Compute the spatial derivative of the value function (dV/dx)
                spatial_derivative(left, right, axis, vf, self.grid, *idxs)

                # do for both left/right derivatives
                for deriv in (left, right):
                    with hcl.if_(deriv.v < dv_min[axis]):
                        dv_min[axis] = deriv.v
                    with hcl.if_(dv_max[axis] < deriv.v):
                        dv_max[axis] = deriv.v
                
                dv[axis] = (left.v + right.v) / 2
                
                dv_diff[idxs + (axis,)] = right.v - left.v

            # Use the model's methods to solve optimal control
            self.model.opt_ctrl(u, dv, t, x)
            self.model.opt_dstb(d, dv, t, x)

            # Calculate dynamical rates of changes
            self.model.dynamics(dx, t, x, u, d)

            # Calculate Hamiltonian terms
            h[idxs] = -hcl_math.dot(dv, dx)

        hcl.mutate(vf.shape, body, name='Hamiltonian')

    def dissipation_stage(self, h, t, xs, *,
                          dv_min, dv_max, dv_diff, 
                          max_alpha):
        """Calculate the dissipation"""

        def body(*idxs):

            x = hcl.compute(self.state_shape, lambda _: 0, name='x')

            # Each has a combination of lower/upper bound on control and disturbance
            dx_ll = hcl.compute(self.state_shape, lambda _: 0, name='dx_ll')
            dx_lu = hcl.compute(self.state_shape, lambda _: 0, name='dx_lu')
            dx_ul = hcl.compute(self.state_shape, lambda _: 0, name='dx_ul')
            dx_uu = hcl.compute(self.state_shape, lambda _: 0, name='dx_uu')

            lower_opt_ctrl = hcl.compute(self.ctrl_shape, lambda _: 0, name='lower_opt_ctrl')
            upper_opt_ctrl = hcl.compute(self.ctrl_shape, lambda _: 0, name='upper_opt_ctrl')
            lower_opt_dstb = hcl.compute(self.dstb_shape, lambda _: 0, name='lower_opt_dstb')
            upper_opt_dstb = hcl.compute(self.dstb_shape, lambda _: 0, name='upper_opt_dstb')

            for n, i in enumerate(idxs):
                x[n] = xs[n][i]

            # Find LOWER BOUND optimal disturbance
            self.model.opt_dstb(lower_opt_dstb, dv_min, t, x)

            # Find UPPER BOUND optimal disturbance
            self.model.opt_dstb(upper_opt_dstb, dv_max, t, x)

            # Find LOWER BOUND optimal control
            self.model.opt_ctrl(lower_opt_ctrl, dv_min, t, x)

            # Find UPPER BOUND optimal control
            self.model.opt_ctrl(upper_opt_ctrl, dv_max, t, x)

            # Find magnitude of rates of changes
            self.model.dynamics(dx_ll, t, x, lower_opt_ctrl, lower_opt_dstb)
            hcl.update(dx_ll, lambda i: hcl_math.abs(dx_ll[i]))

            self.model.dynamics(dx_lu, t, x, lower_opt_ctrl, upper_opt_dstb)
            hcl.update(dx_lu, lambda i: hcl_math.abs(dx_lu[i]))

            self.model.dynamics(dx_ul, t, x, upper_opt_ctrl, lower_opt_dstb)
            hcl.update(dx_ul, lambda i: hcl_math.abs(dx_ul[i]))

            self.model.dynamics(dx_uu, t, x, upper_opt_ctrl, upper_opt_dstb)
            hcl.update(dx_uu, lambda i: hcl_math.abs(dx_uu[i]))

            # Calulate alpha
            alpha = hcl.compute(self.state_shape, 
                                lambda i: hcl_math.max(dx_ll[i], dx_lu[i], 
                                                       dx_ul[i], dx_uu[i]), 
                                name='alpha')

            hcl.update(max_alpha, lambda i: hcl_math.max(max_alpha[i], alpha[i]))

            # Finally we update the hamiltonian
            # dv_diff has shape <grid...> x <states>. Here we use dv_diff at the current grid point.
            dv_diff_here = hcl.compute(self.state_shape, lambda n: dv_diff[idxs + (n,)])
            dissipation_v = hcl_math.dot(dv_diff_here, alpha) / 2
            h[idxs] = -(h[idxs] - dissipation_v)

        hcl.mutate(h.shape, body, name='Dissipation')

    def time_step(self, t, *, 
                  max_alpha):

        step_bound = hcl.scalar(0, 'step_bound')

        tmp = hcl.scalar(0)
        for i, res in enumerate(self.grid.dx):
            tmp.v += max_alpha[i] / res
        step_bound.v = 0.8 / tmp.v

        with hcl.if_(t[1] - t[0] < step_bound.v):
            step_bound.v = t[1] - t[0]

        t[0] += step_bound.v

        return step_bound.v


class HJSolver(Solver):

    def __init__(self, grid, tau, model, **kwargs):
        

        super().__init__(grid, model, **kwargs)

    def __call__(self, tau, *,
                 target, target_mode='min',
                 constraint=None, constraint_mode='max'):

        target_invariant = target.shape == self.grid.shape
        assert target_invariant or target.shape == self.grid.shape + tau.shape
        assert target_mode in ('max', 'min')

        if constraint is not None:
            constraint_invariant = constraint.shape == self.grid.shape
            assert constraint_invariant or constraint.shape == self.grid.shape + tau.shape
            assert constraint_mode in ('max', 'min')

        # Tensor input to our computation graph
        vf = target if target_invariant else target[..., -1]
        t = np.flip(tau)
        xs = [ax.flatten() for ax in self.grid.vs]

        # Extend over time axis
        out = np.zeros(vf.shape + (len(t),))
        out[..., -1] = vf

        ################ USE THE EXECUTABLE ############
        # Variables used for timing
        execution_time = 0
        now = t[-1]

        if self.interactive:
            print("Running...")
            line_length = 0

        # Backward reachable set/tube will be computed over the specified time horizon
        # Or until convergent ( which ever happens first )
        for i in reversed(range(0, len(t)-1)):

            vf = out[..., i+1].copy()

            pde_args = [vf, np.array([now, t[i]]), *xs]
            pde_args = list(map(hcl.asarray, pde_args))

            if self.debug:
                h = hcl.asarray(np.zeros_like(vf))
                pde_args = [h] + pde_args

            while now <= t[i] - 1e-4:

                # Start timing
                start = time.time()

                # Run the execution and pass input into graph
                self._executable(*pde_args)

                # End timing
                end = time.time()

                # Calculate computation time
                execution_time += end - start

                # Get current time from within solver
                if self.debug:
                    now = pde_args[2].asnumpy()[0]
                else:
                    now = pde_args[1].asnumpy()[0]

                # Some information printing
                if self.interactive:
                    line = f"> [{execution_time:.2f}: {i}] Integration time: {end - start:.5f} s - Now: {now:.5f}"
                    line_length = max(line_length, len(line))
                    print(line, end="\r", flush=True)

            if self.debug:
                h = pde_args[0].asnumpy()
                vf = pde_args[1].asnumpy()
            else:
                vf = pde_args[0].asnumpy()

            op = np.minimum if target_mode == 'min' else np.maximum
            vf = op(vf, target) if target_invariant else op(vf, target[..., i])

            if constraint is not None:
                op = np.minimum if constraint_mode == 'min' else np.maximum
                vf = op(vf, -constraint) if constraint_invariant else op(vf, -constraint[..., i])

            out[..., i] = vf

        # Time info printing
        if self.interactive:
            line = f"> Total kernel time: {execution_time:.2f} s"
            line = line.ljust(line_length)
            print(line, end="\n\n")

        # Flip time axis so that earliest time is first
        # out = np.flip(out, axis=-1)

        return out
