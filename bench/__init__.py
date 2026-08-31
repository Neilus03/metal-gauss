"""Benchmark harnesses.

This file exists so that `bench` resolves to this package rather than to
bench/bench.py. A directory without __init__.py is only a namespace portion,
and a namespace portion loses to a regular module of the same name found
anywhere on sys.path -- so `python bench/nerf_synthetic_sweep.py`, which puts
bench/ (containing bench.py) on the path, imported bench.py as `bench` and
failed with "'bench' is not a package".
"""
