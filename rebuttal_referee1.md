# Response to Referee 1

We thank the referee for the careful reading of both the manuscript and the
accompanying software, and for the encouraging assessment that "OptimizedDP
meaningfully advances the state of the art." We are especially grateful for the
concrete, reproducible issues raised about the codebase — they were easy to act
on and have measurably improved the package. Below we respond to each comment in
turn. Comments are quoted in *italics*; our response and the corresponding code
changes follow.

A short summary of the software changes made in response to this review:

* Added a proper, automated **test suite** (`tests/`, run with `pytest`) with
  numerical regression tests and mathematical-property tests for the HJ, TTR and value-iteration solvers, plus unit tests for the grid and shape utilities, and a **GitHub Actions CI** workflow (`.github/workflows/tests.yml`).
* Fixed the `NameError: name 'secondOrderX' is not defined` crash in the 1-D
  solver.
* Fixed a bug in value iteration that prevented multi-dimensional (e.g. `(v, w)`)
  action spaces from working, and rewrote the outdated `odp/MDP_example/`
  files against the current API.
* Fixed a figure-saving bug in `plot_isosurface` and made the example scripts
  run to completion.
* Corrected the `odp/dynamics/__init__.py`, the installation notes
  (`libtinfo5`/`libtinfo6`), and several README inaccuracies; made the `dims`
  argument to `Grid` optional (inferred from the bounds).

---

## Major comments

### 1. Lack of tests; `plotting_examples.py` crashes with `NameError: secondOrderX`

> *The package does not appear to have any tests ... Tests are crucial for
> maintenance of any open source package ... Moreover, I could not get all of the
> examples to run. For instance `plotting_examples.py` failed after several
> minutes with `NameError: name 'secondOrderX' is not defined`.*

We fully agree that automated tests are essential, and we thank the referee for
holding the package to this standard.

**Tests.** We have added a dedicated, automated test suite under `tests/`, run
with `pytest`. It is deliberately fast (all problems use small grids; the full
suite runs in a few seconds) so that it can be executed routinely and in CI. It
contains two complementary kinds of tests:

* **Numerical regression tests** that compare freshly computed value functions
  against small golden reference solutions (stored in `tests/data/`) for the HJ
  reachability solver (1-D/2-D/3-D), the time-to-reach solver, and MDP value
  iteration (2-D scalar-action and 3-D multi-dimensional-action).
* **Mathematical-property tests** that check invariants independent of any stored
  reference — e.g. that a backward reachable tube can only grow, that
  time-to-reach is non-negative and vanishes on the target, and that the
  pendulum value function is non-positive with its maximum at the upright state —
  together with unit tests for the `Grid` and shape-function utilities.

The problem definitions are shared between the tests and the reference-data
generator (`tests/reference_problems.py`), so the tested code and the reference
data cannot silently drift apart. We also added a **continuous-integration
workflow** (`.github/workflows/tests.yml`) that creates the conda environment and
runs the suite on every push and pull request, and a `tests/README.md`
describing how to run and regenerate everything.

**The `secondOrderX` crash.** This was a genuine bug: the 1-D compute graph
called a function named `secondOrderX`, which does not exist — the correct
second-order ENO routine is `secondOrder_ENO1D_X0`. Because the default accuracy
is `"medium"`, *any* 1-D solve crashed, which is exactly what the referee hit at
the end of `plotting_examples.py`. We have corrected both occurrences in
`odp/computeGraphs/graph_1D.py`, and the 1-D path is now exercised by the test
suite (`tests/test_hj_solver.py`). We separately found and fixed a figure-saving
bug (`plot_isosurface` called `write_image` on an extension-less filename even in
interactive-HTML mode; see Minor comments below) and reduced the very large
default grids in `plotting_examples.py` so that the full example now runs to
completion in a reasonable time.

We agree with the referee's broader point that examples are not a substitute for
tests, and we now treat them as separate concerns: the examples are for
illustration, and the `tests/` suite is for verification.

### 2. Documentation is limited; `odp/MDP_example/Example_3D.py` is outdated and incorrect

> *... if one searches for documentation, they might find
> `odp/MDP_example/Example_3D.py`, which appears to be outdated and incorrect!
> Outdated documentation must be removed and some correct documentation
> provided.*

The referee is correct. The `odp/MDP_example/` files documented an *older* value-
iteration interface (`transition(sVals, action, bounds, trans, goal)` with the
problem defined through class attributes) that no longer matches the maintained
API (`solveValueIteration` with `transition(sVals, iVals, u)` returning the
successor states). We have:

* **Rewritten `Example_3D.py`** into a correct, self-contained, runnable example
  against the current API, with a documentation header that precisely describes
  the required interface (`maxTransitions`, `transition`, `reward`, and the
  meaning of each argument). This example is now itself covered by the test suite
  (`tests/test_mdp.py::test_mdp_multidimensional_action_space`).
* **Rewritten `Example_4D/5D/6D.py`** against the same current API; each runs to
  completion and produces a correctly propagated value function.
* **Removed `Example_7D.py`.** The maintained value-iteration entry point
  (`solveValueIteration`) currently supports 2-D through 6-D; the 7-D file relied
  on a separate, legacy code path and would have been exactly the kind of
  outdated/incorrect documentation the referee (rightly) objects to. We would
  rather ship correct examples for the supported dimensions than a misleading one.

While addressing this comment we also uncovered and fixed a substantive bug (see
below) that had made the 3-D example incorrect.

### 3. The "memory increases linearly with dimension" statement is confusing

> *"As we increase the number of dimensions, the DRAM memory required for these
> temporary array goes up linearly." ... each of these arrays are the size of the
> grid, which increases exponentially, so I am not sure why the linear increase
> is important.*

We agree this sentence is unclear and thank the referee for flagging it; the word
"linearly" was describing the wrong quantity. The intended point concerns the
*number of temporary grid-sized arrays*, not the total memory:

* A grid-based HJ update must, at each point, form upwind spatial derivatives in
  every dimension. A conventional (e.g. Level-Set-Toolbox-style) implementation
  materializes a **separate temporary array, each the size of the full grid, for
  every dimension** (and for several intermediate quantities). Concretely, in our
  own 6-D graph the per-dimension difference terms are `deriv_diff1 … deriv_diff6`
  — i.e. *D* grid-sized temporaries for a *D*-dimensional problem.
* Therefore the temporary storage is `(number of dimensions) × (size of one
  grid) = D · N^D`. The **base grid `N^D` is of course exponential in `D` and is
  unavoidable for any dense grid method** (we do not claim otherwise). What we
  were trying to highlight is the **linear-in-`D` multiplicative overhead** that
  sits on top of that base: the temporary footprint is a factor of roughly `D`
  larger than the value-function array itself.
* OptimizedDP reduces this overhead by computing the left/right derivatives as
  *scalar, per-grid-point* quantities inside a fused kernel rather than
  allocating and retaining full grids of intermediates. This does not (and cannot)
  change the exponential `N^D` base; it removes the linear-in-`D` *multiplier* on
  it, which is what makes the difference between an 8-D problem fitting or not
  fitting in a fixed DRAM budget.

**Proposed replacement text for the manuscript** (to replace the ambiguous
sentence wherever it appears):

> "For a `D`-dimensional problem a conventional grid-based implementation
> allocates one grid-sized temporary array per dimension to hold the upwind
> spatial-derivative terms, so the temporary memory is approximately `D · N^D` —
> that is, a factor of `D` on top of the `N^D` storage of the value function
> itself. While the `N^D` term is exponential in the dimension and is inherent to
> any dense-grid method, the additional factor of `D` is not: OptimizedDP computes
> these derivatives as scalar per-grid-point quantities within a single fused
> kernel, eliminating the `D` grid-sized temporaries. The temporary-memory
> *overhead* relative to the value-function array is thus reduced from linear in
> the dimension to essentially constant, which is what allows problems such as the
> 8-D example to fit within a fixed DRAM budget."

We will also revise the caption/discussion of Figure 2 so that it makes explicit
that each of the "N arrays" is grid-sized, and that the contribution is the
elimination of these `D` grid-sized temporaries rather than any change to the
exponential base.

---

## Minor comments

### Installation: `libtinfo5` is deprecated in favour of `libtinfo6`

> *Installation requires manual installation of an old library, `libtinfo5`. This
> seems to be deprecated in favor of `libtinfo6`.*

Thank you — the `libtinfo.so.5` requirement comes from the pinned HeteroCL 0.3
runtime. We have updated the README to explain this and to instruct users on
newer distributions (where `libtinfo5` has been dropped) to install `libtinfo6`
instead, and the CI workflow now installs `libtinfo5` with an automatic fallback
to `libtinfo6`.

> **TODO (internal — resolve/remove before submission):** Verify the `libtinfo6`
> path end-to-end in a genuinely `libtinfo5`-free environment (a clean
> `ubuntu:24.04` container or a CI run on `ubuntu-24.04`). So far it has only been
> confirmed indirectly, via a `libtinfo.so.5` → `libtinfo.so.6` symlink: `apt
> install libtinfo6` provides `libtinfo.so.6`, **not** the `libtinfo.so.5` soname
> HeteroCL loads, so `libtinfo6` works only once that compatibility symlink
> exists. The README now documents the symlink (env-local or system-wide) plus a
> `python -c "import heterocl"` check; the CI `libtinfo6` fallback still needs the
> symlink step and that import smoke check added before this paragraph is
> accurate.

### `odp/dynamics/__init__.py` is incorrect (`DubinsCar5D.py` does not exist)

> *The `odp/dynamics/__init__.py` file appears to be incorrect because, at
> minimum, `DubinsCar5D.py` does not exist.*

Corrected. The `__init__.py` now imports only modules that exist (the 5-D system
is `DubinsCar5DAvoid`), and we removed a stray duplicate import. Importing the
`odp.dynamics` package is also now implicitly exercised by the test suite.

### README questions

> *Why is it necessary to provide a `dims` argument? Can't this be inferred from
> the grid arguments in the constructor?*

Agreed — `dims` was redundant with `len(grid_min)`. We have made it **optional**:
if omitted it is inferred from the bounds, and if supplied it is validated against
them as a sanity check (so existing code that passes it continues to work). This
is covered by `tests/test_grid.py::test_grid_infers_dims_when_omitted`.

> *Do `grid_min` and `grid_max` need to be numpy arrays or will lists work?*

Plain Python lists work — they are converted to numpy arrays inside the
constructor. We now state this in the README and added a regression test
(`tests/test_grid.py::test_grid_accepts_plain_python_lists`).

> *What is `pd`?*

`pd` is the `periodicDims` argument: the list of 0-indexed dimensions that are
periodic (e.g. an angle in `[-pi, pi)`). We have renamed/annotated it in the
README example and documented its meaning.

> *`examples/plotting_example.py` link is dead.*

Fixed. The file is `examples/plotting_examples.py` (plural); we corrected this and
the other stale example links in the README to valid `blob/master` URLs.

---

## Additional comments (approach / design)

We appreciate these thoughtful comments on the interface. They do not correspond
to defects, but we address each and note where we have improved the documentation
or the code.

> *The user defining the dynamics needs to write HeteroCL code, e.g.
> `hcl.scalar` ... we should consider using languages designed for numerical
> computing in the future.*

We agree this is a real ergonomic trade-off. Exposing HeteroCL in the dynamics
specification is what lets the same user-written dynamics be compiled into the
fused, parallelized kernel that gives the performance reported in the paper; a
pure-NumPy specification would have to be interpreted per grid point. We have
clarified this trade-off in the documentation. A lighter-weight (e.g.
NumPy/JAX-like) front-end that is lowered to the same kernel is a natural avenue
for future work, and we now say so explicitly in the paper's limitations
discussion.

> *The HJ-PDE interface requires the user to implement `opt_ctrl` ... it seems
> like the package should automatically calculate the optimal control.*

This is a fair observation. The optimal control is the arg-extremum of the
Hamiltonian, `u* = argmin/max_u <spatial_deriv, f(x,u)>`. For the common
control-affine case this has a closed form (bang-bang in each input), which is why
libraries in this family — including helperOC, as the referee notes — ask the user
to supply it once; doing so also lets the user encode input constraints and
control-sharing structure directly. Computing it automatically in general requires
either symbolic differentiation of user dynamics or a numerical inner optimization
*at every grid point and every timestep*, which can dominate runtime. We have
added a note documenting this rationale, and we agree that providing an
automatic default for the control-affine case (the majority of examples) is a
worthwhile future extension.

> *The `transition` function returns a matrix mixing probabilities with state
> values, which ... limits the support of the transition distribution to a finite
> set, precluding distributions such as Gaussians.*

We agree the mixed probability/state layout is not the most transparent
representation, and we have documented it more carefully (each row is one possible
outcome: column 0 its probability, the remaining columns its successor state). We
note that grid-based value iteration fundamentally operates on a discretized state
space, so a continuous transition kernel (e.g. a Gaussian) is in practice
represented by its probability mass on a finite set of neighbouring grid cells —
which is exactly what a multi-row transition matrix expresses. A continuous
distribution is therefore supported by discretizing it onto grid cells (increasing
`maxTransitions`). Separating the probabilities and states into two returned
arrays is a clean interface improvement we are happy to make in a future revision.

> *The `transition` function takes an `iVals` argument ... interfaces involving
> both the state space and the index space seem prone to errors ... better to have
> the user write code entirely in state values OR index values.*

We agree, and we suspect the previous, undocumented interface was the source of
the confusion. In the current API the user writes `transition`/`reward` **entirely
in terms of continuous state values** (`sVals`): the successor states returned are
state values, not indices, and all of our examples ignore `iVals` entirely.
`iVals` is passed only as an optional convenience (the integer grid indices of the
current state) for users who want it; it can be — and normally is — ignored. We
have documented this explicitly at the top of `Example_3D.py`, so that the
recommended usage is unambiguously state-value-based, matching the referee's
preference. (While clarifying this we also fixed a bug whereby multi-dimensional
action spaces did not work at all in 2-D–5-D value iteration; the internal
per-action buffer was mis-sized, which is now corrected and regression-tested.)

---

We believe these changes address every issue the referee raised — in particular
the two that motivated the recommendation, namely the absence of tests and the
non-running/outdated examples — and we thank the referee again for feedback that
has made both the paper and the software substantially stronger.
