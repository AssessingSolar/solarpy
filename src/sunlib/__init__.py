from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover
    __version__ = version(__package__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"


# Import of functions that should be accessible from the package top-level
# from .layout import generate_field_layout  # noqa: F401
