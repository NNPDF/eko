# -*- coding: utf-8 -*-
"""
This module contain the possible scale variations integrals.
"""
from enum import Enum

from . import expanded, exponentiated


class Modes(Enum):
    """Scale Variation modes"""

    unvaried = 0
    exponentiated = 1
    expanded = 2
