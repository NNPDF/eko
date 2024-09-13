"""EKO database configuration."""

from typing import Union

from eko.io.runcards import Legacy, OperatorCard, TheoryCard
from eko.io.types import RawCard


def update_runcards(
    theory: Union[RawCard, TheoryCard], operator: Union[RawCard, OperatorCard]
):
    """Update legacy runcards.

    This function is mainly defined for compatibility with the old interface.
    Prefer direct usage of :class:`Legacy` in new code.

    Consecutive applications of this function yield identical results::

        cards = update(theory, operator)
        assert update(*cards) == cards
    """
    if isinstance(theory, TheoryCard) or isinstance(operator, OperatorCard):
        # if one is not a dict, both have to be new cards
        assert isinstance(theory, TheoryCard)
        assert isinstance(operator, OperatorCard)
        return theory, operator

    cards = Legacy(theory, operator)
    return cards.new_theory, cards.new_operator
