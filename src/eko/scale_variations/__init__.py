"""Define possible scale variations schemes.

A Mathematica snippet to check the formulas is available in the extras
folder.
"""

import enum
from typing import Any, Dict

from ..io.types import ScaleVariationsMethod


class Modes(enum.IntEnum):
    """Enumerate scale variation modes."""

    unvaried = enum.auto()
    exponentiated = enum.auto()
    expanded = enum.auto()


def sv_mode(s: ScaleVariationsMethod) -> Modes:
    """Return the scale variation mode.

    Parameters
    ----------
    s :
        string representation

    Returns
    -------
    i :
        int representation
    """
    if s is not None:
        return Modes[s.value]
    return Modes.unvaried


class ScaleVariationModeMixin:
    """Mixin to cast scale variation mode."""

    config: Dict[str, Any]

    @property
    def sv_mode(self) -> Modes:
        """Return the scale variation mode."""
        return sv_mode(self.config["ModSV"])
