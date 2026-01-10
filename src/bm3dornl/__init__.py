"""
Contains the entry point for the application
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bm3dornl")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

from .bm3d import bm3d_ring_artifact_removal  # noqa: F401
