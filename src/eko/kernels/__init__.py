"""The solutions to the |DGLAP| equations."""

import enum

from ..io.types import EvolutionMethod


class EvoMethods(enum.IntEnum):
    """Enumerate evolution methods."""

    ITERATE_EXACT = enum.auto()
    ITERATE_EXPANDED = enum.auto()
    PERTURBATIVE_EXACT = enum.auto()
    PERTURBATIVE_EXPANDED = enum.auto()
    TRUNCATED = enum.auto()
    ORDERED_TRUNCATED = enum.auto()
    DECOMPOSE_EXACT = enum.auto()
    DECOMPOSE_EXPANDED = enum.auto()


def ev_method(s: EvolutionMethod) -> EvoMethods:
    """Return the evolution method.

    Parameters
    ----------
    s :
        string representation

    Returns
    -------
    i :
        int representation
    """
    return EvoMethods[s.value.upper().replace("-", "_")]
