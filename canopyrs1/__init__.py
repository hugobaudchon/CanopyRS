"""CanopyRS: detect, segment and classify trees in high-resolution geospatial imagery.

This is the rewrite of CanopyRS, with geodataset merged in. It is called `canopyrs1` until it
replaces `canopyrs` (see MIGRATION_PLAN.md at the repository root).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("CanopyRS")   # the installed distribution, shared with `canopyrs`
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
