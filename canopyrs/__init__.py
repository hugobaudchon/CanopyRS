from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("CanopyRS")  # distribution name from [project].name
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]