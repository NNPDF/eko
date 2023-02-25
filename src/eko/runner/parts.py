"""Operator components."""
from abc import ABC
from dataclasses import dataclass

from ..io.dictlike import DictLike


@dataclass
class Part(DictLike, ABC):
    """An atomic operator ingredient."""

    name: str
