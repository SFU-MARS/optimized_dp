# OptimizedDP test suite

Automated tests for the OptimizedDP solvers and utilities. They are designed to
run in a few seconds on a laptop (all problems use small grids) so they can be
run routinely and in continuous integration.

## Running the tests

From the repository root, inside the `odp` conda environment:

```bash
pip install pytest      # once, if not already installed
python -m pytest
```

## What is covered

| File                      | Scope                                                        |
|---------------------------|--------------------------------------------------------------|
| `test_grid.py`            | `Grid` construction: dx/points, list-vs-array inputs, dimension checks, periodic dimensions |
| `test_shapes.py`          | Implicit-surface helpers (`CylinderShape`, `ShapeRectangle`) |
| `test_hj_solver.py`       | Hamilton-Jacobi reachability (`HJSolver`) in 1D/2D/3D        |
| `test_ttr_solver.py`      | Time-to-reach computation (`TTRSolver`)                      |
| `test_mdp.py`             | MDP value iteration (`solveValueIteration`)                  |

Each solver has two kinds of tests:

1. **Regression tests** compare a freshly computed value function against a
   golden reference array stored in `tests/data/`.
2. **Property tests** assert mathematical/physical invariants (e.g. a backward
   reachable tube can only grow, time-to-reach is non-negative and zero at the
   target, the pendulum value function is non-positive with a maximum at the
   upright state).

The problem definitions themselves live in `reference_problems.py` and are
shared between the tests and the reference-data generator, so the two can never
drift apart.

## Regenerating reference data

Only after an *intentional* change to a numerical algorithm, and once the new
output has been verified as correct:

```bash
python tests/regenerate_references.py
```

This overwrites the `.npy` files in `tests/data/` (a few kilobytes each).
