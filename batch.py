"""Headless batch runner — entry point kept here for CLI use.

The implementation lives in correlative_imaging/batch.py so it is
importable regardless of whether the package is installed or run from source.
"""
from correlative_imaging.batch import BatchRunner, ProgressFn  # noqa: F401
