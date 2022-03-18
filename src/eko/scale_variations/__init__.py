# -*- coding: utf-8 -*-
"""
This module contain the possible scale variations integrals.
"""
import enum

from . import expanded, exponentiated


class Modes(enum.IntEnum):
    """Scale Variation modes"""

    unvaried = enum.auto()
    exponentiated = enum.auto()
    expanded = enum.auto()
