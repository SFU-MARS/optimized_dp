# OptimizedDP — Referee 1: Questions and Code Changes

This document summarizes, for our paper collaborators, the concerns raised by
Referee 1 and the concrete changes we made to the OptimizedDP package in
response. Referee 1's overall recommendation was **Reject**, while noting that
"OptimizedDP meaningfully advances the state of the art" (the first 8-dimensional
HJ-PDE problem they had seen solved). The two concerns that drove the
recommendation were (1) the absence of an automated test suite and (2) a
non-running / outdated example; both are now fully addressed in code.

A quick map of the review structure: **3 major comments**, **several minor
comments** (installation + README), and **4 design comments** (interface
opinions that do not require code changes but which we address in the paper /
documentation).

---

## Referee 1's Questions

### Major comments

**M1 — No tests; an example crashes.**
> The package does not appear to have any tests (`test_mdp_3d.py` does not work).
> Tests are crucial for maintenance of any open-source package and should be a
> requirement for publication. Examples are poor substitutes for tests (slow, not
> automated). Moreover, `plotting_examples.py` failed after several minutes with
> `NameError: name 'secondOrderX' is not defined`.

**M2 — Limited documentation; an outdated, incorrect example.**
> Documentation is limited to the README and examples. Worse, a user searching
> for documentation may find `odp/MDP_example/Example_3D.py`, which is outdated
> and incorrect. Outdated documentation must be removed and correct documentation
> provided.

**M3 — The "memory increases linearly with dimension" statement is confusing.**
> The paper repeatedly states "the DRAM memory required for these temporary
> arrays goes up linearly." But with a fixed number of divisions per dimension
> the number of grid elements grows *exponentially* with dimension. Figure 2
> shows "N arrays," which implies linearly more arrays, but each array is
> grid-sized (exponential), so why is the linear increase important? This must be
> explained more clearly.

### Minor comments — installation and running

**m1 — `libtinfo5` is deprecated.** Installation requires manually installing the
old `libtinfo5` library, which appears to be deprecated in favor of `libtinfo6`.

**m2 — `odp/dynamics/__init__.py` is incorrect.** At minimum it references
`DubinsCar5D.py`, which does not exist.

### Minor comments — README

**m3 — Why require a `dims` argument?** Can't it be inferred from the grid
arguments in the constructor?

**m4 — Do `grid_min` / `grid_max` have to be numpy arrays, or will lists work?**

**m5 — What is `pd`?**

**m6 — The `examples/plotting_example.py` link is dead.**

### Design comments (interface opinions; not required before publication)

**d1 — Writing dynamics requires HeteroCL.** The user must write HeteroCL code
(e.g. `hcl.scalar`) rather than plain Python/NumPy.

**d2 — The user must implement `opt_ctrl`.** It seems the package should compute
the optimal control automatically, at least for simple cases.

**d3 — The `transition` matrix mixes probabilities and states.** This is
confusing and limits support to finite-support distributions, precluding e.g.
Gaussians.

**d4 — The `transition` function takes an `iVals` (index-space) argument.**
Mixing state space and index space is error-prone; the user should write code
entirely in state values *or* entirely in index values.

---

## Code Changes Made

### For M1 — a real, automated test suite + CI

* Added **`tests/`**, run with a single `python -m pytest` from the repo root.
  All problems use small grids, so the full suite runs in a few seconds and is
  suitable for routine and CI use.
* Two complementary kinds of tests for each solver:
  * **Regression tests** compare a freshly computed value function against a
    small golden reference array stored in `tests/data/` (HJ 1D/2D/3D, TTR,
    MDP 2D scalar-action and 3D multi-dimensional-action).
  * **Property tests** assert mathematical invariants with no stored reference:
    a backward reachable tube can only grow; time-to-reach is non-negative and
    zero at the target; the pendulum value function is non-positive with its
    maximum at the upright state. Plus unit tests for `Grid` and the shape
    helpers.
* **Shared problem definitions** (`tests/reference_problems.py`) are used by
  *both* the tests and the reference-data generator
  (`tests/regenerate_references.py`), so the tested code and reference data
  cannot silently drift apart.
* **Zero-config discovery**: `pytest.ini` sets the test path and options;
  `conftest.py` ensures the local checkout of `odp` wins on `sys.path`.
* **Continuous integration**: `.github/workflows/tests.yml` builds the conda
  environment and runs the suite on every push and pull request (with a
  `libtinfo5` → `libtinfo6` fallback). A `tests/README.md` documents how to run
  and regenerate everything.

### For M1 — the `secondOrderX` crash

* Fixed a genuine bug in `odp/computeGraphs/graph_1D.py`: the 1-D compute graph
  called a nonexistent function `secondOrderX`; the correct second-order ENO
  routine is `secondOrder_ENO1D_X0`. Because the default accuracy is `"medium"`,
  *every* 1-D solve crashed — exactly what the referee hit at the end of
  `plotting_examples.py`. Both occurrences are fixed and the 1-D path is now
  exercised by `tests/test_hj_solver.py`.
* Separately fixed a figure-saving bug in `odp/Plots/plotting_utilities.py`
  (`plot_isosurface` called `write_image` on an extension-less filename even in
  interactive-HTML mode) and reduced the oversized default grids in
  `examples/plotting_examples.py` so the example now runs to completion.

### For M2 — correct, current MDP examples

* **Rewrote `odp/MDP_example/Example_3D.py`** against the maintained API
  (`solveValueIteration` with `transition(sVals, iVals, u)`), with a
  documentation header that precisely describes the required interface
  (`maxTransitions`, `transition`, `reward`, and the meaning of each argument).
  It is now covered by `tests/test_mdp.py::test_mdp_multidimensional_action_space`.
* **Rewrote `Example_4D/5D/6D.py`** against the same current API; each runs to
  completion and produces a correctly propagated value function.
* **Removed `Example_7D.py`.** `solveValueIteration` currently supports 2-D
  through 6-D; the 7-D file relied on a legacy code path and would itself have
  been the kind of outdated/incorrect documentation the referee objected to.
* While rewriting these examples we found and fixed a substantive bug (see the
  d4 note) that had made the 3-D example incorrect.

### For M3 — the memory statement (manuscript text, no code change)

* The word "linearly" described the wrong quantity. The intended point: a
  conventional grid-based implementation allocates *one grid-sized temporary
  array per dimension* for the upwind spatial-derivative terms, so temporary
  memory is approximately `D · N^D`. The `N^D` base is exponential and
  unavoidable for any dense-grid method; the additional **factor of `D`** is not.
  OptimizedDP computes these derivatives as **scalar, per-grid-point** quantities
  inside a fused kernel, eliminating the `D` grid-sized temporaries — reducing
  the *overhead relative to the value-function array* from linear-in-`D` to
  essentially constant. This is what lets, e.g., the 8-D problem fit within a
  fixed DRAM budget.
* We provide replacement text for the ambiguous sentence and will revise the
  Figure 2 caption to make explicit that each of the "N arrays" is grid-sized
  and that the contribution is eliminating those `D` grid-sized temporaries — not
  changing the exponential base.

### For the minor comments

* **m1 (`libtinfo5`)**: The requirement comes from the pinned HeteroCL 0.3
  runtime. Updated the README to explain this and to tell users on newer
  distributions to install `libtinfo6`; CI installs `libtinfo5` with an automatic
  `libtinfo6` fallback.
* **m2 (`__init__.py`)**: `odp/dynamics/__init__.py` now imports only modules
  that exist (the 5-D system is `DubinsCar5DAvoid`) and a stray duplicate import
  was removed. Importing `odp.dynamics` is now exercised by the test suite.
* **m3 (`dims`)**: Made `dims` **optional** in `odp/Grid/GridProcessing.py`:
  inferred from the bounds when omitted, validated against them when supplied
  (so existing callers still work). Covered by
  `tests/test_grid.py::test_grid_infers_dims_when_omitted`.
* **m4 (lists vs arrays)**: Plain Python lists work — they are converted to numpy
  arrays in the constructor. Documented in the README; regression-tested by
  `tests/test_grid.py::test_grid_accepts_plain_python_lists`.
* **m5 (`pd`)**: `pd` is the `periodicDims` argument — the 0-indexed dimensions
  that are periodic (e.g. an angle in `[-pi, pi)`). Renamed/annotated and
  documented in the README example.
* **m6 (dead link)**: The file is `examples/plotting_examples.py` (plural);
  corrected this and other stale example links in the README to valid
  `blob/master` URLs.

### For the design comments

* **d1 (HeteroCL)**: Documented the trade-off — exposing HeteroCL is what lets
  user-written dynamics compile into the fused, parallelized kernel that delivers
  the paper's performance; a pure-NumPy spec would be interpreted per grid point.
  A lighter NumPy/JAX-like front-end lowered to the same kernel is noted as
  future work in the limitations discussion.
* **d2 (`opt_ctrl`)**: Documented the rationale — the optimal control is
  `u* = argmin/max_u <spatial_deriv, f(x,u)>`, which for the common
  control-affine case is closed-form (bang-bang) and is supplied once by the
  user (as in helperOC); it also lets users encode input constraints and
  control-sharing. Automatic computation in general needs symbolic
  differentiation or a numerical inner optimization at every grid point and
  timestep. We note an automatic control-affine default as worthwhile future
  work.
* **d3 (transition layout)**: Documented the matrix layout more carefully (each
  row is one outcome: column 0 its probability, remaining columns its successor
  state) and explained that grid-based value iteration is inherently over a
  discretized state space, so a Gaussian is represented by its probability mass
  on neighbouring cells (increase `maxTransitions`). Splitting probabilities and
  states into two returned arrays is noted as a clean future interface
  improvement.
* **d4 (`iVals`)**: Documented that in the current API the user writes
  `transition`/`reward` **entirely in continuous state values** (`sVals`);
  successor states are state values, not indices, and all examples ignore
  `iVals` (passed only as an optional convenience). **While clarifying this we
  fixed a substantive bug**: multi-dimensional action spaces (e.g. `(v, w)`)
  did not work at all in 2-D–5-D value iteration because the internal
  per-action buffer (`intermeds`) was mis-sized (`actions.shape` instead of
  `actions.shape[0]`). This is corrected and regression-tested by
  `tests/test_mdp.py::test_mdp_multidimensional_action_space`.

---

## Summary table: question → change

| # | Referee 1's question / concern | What we changed | Where |
|---|---|---|---|
| **M1a** | No automated tests | Added a `pytest` suite (regression + property tests) with shared problem definitions and CI | `tests/`, `pytest.ini`, `conftest.py`, `.github/workflows/tests.yml` |
| **M1b** | `plotting_examples.py` crashes with `NameError: secondOrderX` | Fixed the wrong function name (`secondOrderX` → `secondOrder_ENO1D_X0`); 1-D path now tested | `odp/computeGraphs/graph_1D.py`, `tests/test_hj_solver.py` |
| **M1c** | Examples don't run to completion (figure saving, huge grids) | Fixed `plot_isosurface` image-save logic; reduced default grids | `odp/Plots/plotting_utilities.py`, `examples/plotting_examples.py` |
| **M2** | `Example_3D.py` outdated/incorrect; docs limited | Rewrote `Example_3D/4D/5D/6D.py` against the current API with a documented interface; removed unsupported `Example_7D.py` | `odp/MDP_example/*` |
| **M3** | "Memory increases linearly" is confusing | Revised manuscript text + Figure 2 caption: `N^D` base is exponential and unavoidable; the eliminated cost is the `D` grid-sized *temporaries* (linear overhead) | manuscript (text provided) |
| **m1** | `libtinfo5` is deprecated | README note + CI `libtinfo5` → `libtinfo6` fallback | `README.md`, `.github/workflows/tests.yml` |
| **m2** | `dynamics/__init__.py` imports nonexistent `DubinsCar5D` | Import only existing modules (`DubinsCar5DAvoid`); removed duplicate import | `odp/dynamics/__init__.py` |
| **m3** | Why require `dims`? | Made `dims` optional (inferred from bounds, validated if given) | `odp/Grid/GridProcessing.py`, `tests/test_grid.py` |
| **m4** | Lists vs numpy arrays for bounds? | Lists accepted and converted internally; documented + tested | `README.md`, `tests/test_grid.py` |
| **m5** | What is `pd`? | Documented `pd` = `periodicDims` (0-indexed periodic dims) | `README.md` |
| **m6** | Dead `plotting_example.py` link | Fixed link to `plotting_examples.py` and other stale links | `README.md` |
| **d1** | Dynamics require HeteroCL, not plain Python | Documented the performance trade-off; NumPy/JAX front-end noted as future work | docs / paper limitations |
| **d2** | User must implement `opt_ctrl` | Documented the argmax-Hamiltonian rationale; auto control-affine default noted as future work | docs |
| **d3** | `transition` mixes probabilities and states; no Gaussians | Documented the row layout; explained discretized-kernel representation; split-array interface noted as future work | docs |
| **d4** | `transition` mixes state space and index space (`iVals`) | Documented state-value-only usage (`iVals` optional/ignored); **fixed the multi-dimensional action-space bug** found while clarifying | `odp/valueIteration/value_iteration_{2..5}D.py`, `odp/MDP_example/Example_3D.py`, `tests/test_mdp.py` |
