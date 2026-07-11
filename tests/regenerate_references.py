"""
Regenerate the golden reference arrays used by the regression tests.

Run this **only** when you have intentionally changed a numerical algorithm and
have verified that the new output is correct::

    python tests/regenerate_references.py

The generated ``.npy`` files are committed to ``tests/data`` and are small
(a few kilobytes each) by design.
"""

import os
import sys

# Make sure the local repo checkout (and this directory) win on sys.path so the
# reference data is generated from the code in this repository.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from reference_problems import PROBLEMS

DATA_DIR = os.path.join(_HERE, "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, solve in PROBLEMS.items():
        print(f"\n=== Regenerating reference for '{name}' ===")
        result = solve()
        path = os.path.join(DATA_DIR, f"{name}.npy")
        np.save(path, result)
        print(f"Saved {path}  shape={result.shape}")


if __name__ == "__main__":
    main()
