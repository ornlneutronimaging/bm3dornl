"""
Contains the entry point for the application
"""

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    __version__ = "0.0.1"

from .bm3d import bm3d_ring_artifact_removal  # noqa: F401
