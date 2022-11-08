"""Structures to hold runcard information.

.. todo::
    Inherit from :class:`eko.output.struct.DictLike`, to get
    :meth:`eko.output.struct.DictLike.raw` for free and avoid duplication.

"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TheoryCard:
    """Represent theory card content."""

    order: Tuple[int, int]
    """Perturbatiive order tuple, ``(QCD, QED)``."""

    @classmethod
    def load(cls, card: dict):
        """Load from runcard raw content.

        Parameters
        ----------
        card: dict
            content of the theory runcard

        Returns
        -------
        TheoryCard
            the loaded instance

        """
        return cls(order=tuple(card["order"]))


@dataclass
class OperatorCard:
    """Represent operator card content."""

    xgrid: list
    """Grid defining internal interpolation basis."""

    @classmethod
    def load(cls, card: dict):
        """Load from runcard raw content.

        Parameters
        ----------
        card: dict
            content of the theory runcard

        Returns
        -------
        TheoryCard
            the loaded instance

        """
        return cls(xgrid=card["xgrid"])
