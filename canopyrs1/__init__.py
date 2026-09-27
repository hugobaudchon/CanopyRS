"""CanopyRS: detect, segment and classify trees in high-resolution geospatial imagery."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("CanopyRS")   # the installed distribution
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
