"""The solutions to the |DGLAP| equations."""

import enum


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


def ev_method(s: EvoMethods) -> EvoMethods:
    """Return the evolution methods.

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
