"""Module to load the version created by versioningit

Will fall back to a default packagename  is not installed"""

try:
    from ._version import __version__
except ModuleNotFoundError:
    __version__ = "0.0.1"
