"""
Pytest configuration for the OptimizedDP test suite.

Ensures that ``import odp`` resolves to *this* repository checkout (the one the
tests live in) rather than any other copy that may be installed in the active
environment, and makes the shared ``tests/reference_problems`` module
importable from the test files.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Prepend so the local package wins over any editable/site-packages install.
for path in (os.path.join(ROOT, "tests"), ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
