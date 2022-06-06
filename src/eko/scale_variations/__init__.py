# -*- coding: utf-8 -*-
"""This module contain the possible scale variations schemes."""
import enum

from . import expanded, exponentiated


class Modes(enum.IntEnum):
    """Scale Variation modes"""

    unvaried = enum.auto()
    exponentiated = enum.auto()
    expanded = enum.auto()


def sv_mode(s):
    """Returns the scale variation mode.

    Parameters
    ----------
    s : str
        string representation

    Returns
    -------
    enum.IntEnum
        enum representation
    """
    if s is not None:
        return Modes[s]
    return Modes.unvaried


class ModeMixin:
    """Mixin to cast scale variation mode."""

    @property
    def sv_mode(self):
        """Returns the scale variation mode"""
        return sv_mode(self.config["ModSV"])
