"""Shared fixtures for the pytedi.us tests.

These tests need the optional US dependencies (pip install "pytedi[us]"). When they are
missing, as in the core-only CI job, every test in this folder is skipped.
"""
import pytest

for _module in ("geopandas", "rasterio", "shapely", "pysheds"):
    pytest.importorskip(_module, reason='needs the optional US dependencies: pip install "pytedi[us]"')
