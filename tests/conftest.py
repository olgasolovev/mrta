"""!
@file conftest.py
@brief pytest configuration: ensure the repository root is importable so
that ``import mrta``, ``import run_scan_fp`` etc. work when pytest is
invoked from any location.  Also declares the ``--fast`` CLI flag reused
by the standalone validation gate suites.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_addoption(parser):
    parser.addoption("--fast", action="store_true", default=False,
                     help="use reduced grids (legacy gate suite).")
