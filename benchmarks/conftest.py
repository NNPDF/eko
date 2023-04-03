import numpy as np
import pytest

from eko import interpolation
from eko.io.runcards import TheoryCard
from ekobox import cards


@pytest.fixture
def theory_card():
    return cards.example.theory()


@pytest.fixture()
def theory_ffns(theory_card):
    def set_(flavors: int) -> TheoryCard:
        i = flavors - 3
        for q in "cbt"[i:]:
            setattr(theory_card.matching, q, np.inf)
        return theory_card

    return set_


@pytest.fixture
def operator_card():
    card = cards.example.operator()
    card.xgrid = interpolation.XGrid([0.1, 0.3, 0.5, 1.0])
    card.configs.interpolation_polynomial_degree = 2

    return card
