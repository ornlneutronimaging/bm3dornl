"""
Contains the entry point for the application
"""

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    __version__ = "unknown"


def PackageName():  # pylint: disable=invalid-name
    """This is needed for backward compatibility because mantid workbench does "from shiver import Shiver" """
    from .packagenamepy import PackageName as packagename  # pylint: disable=import-outside-toplevel

    return packagename()
