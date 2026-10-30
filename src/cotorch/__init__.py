"""CoTorch: an experimental, early-stage machine learning framework."""

from importlib.metadata import version
from ._core import matmul

__version__ = version("cotorch")

__all__ = ["__version__", "matmul"]