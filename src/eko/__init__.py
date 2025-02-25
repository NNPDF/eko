"""Evolution Kernel Operators."""

from . import io, version
from .io.struct import EKO
from .runner import solve

__version__ = version.__version__

__all__ = [
    "io",
    "EKO",
    "solve",
]
