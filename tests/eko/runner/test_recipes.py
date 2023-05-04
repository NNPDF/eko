from pathlib import Path

import pytest

from eko import EKO
from eko.io.items import Evolution, Matching
from eko.io.runcards import OperatorCard, TheoryCard
from eko.matchings import Atlas
from eko.quantities.heavy_quarks import MatchingScales
from eko.runner.recipes import elements


@pytest.fixture
def neweko(theory_card: TheoryCard, operator_card: OperatorCard, tmp_path: Path):
    path = tmp_path / "eko.tar"
    with EKO.create(path) as builder:
        yield builder.load_cards(theory_card, operator_card).build()

    path.unlink(missing_ok=True)


def test_elements():
    scales = MatchingScales([10.0, 20.0, 30.0])

    atlas = Atlas(scales, (50.0, 5))
    onestep = elements((60.0, 5), atlas)
    assert len(onestep) == 1
    assert isinstance(onestep[0], Evolution)
    assert not onestep[0].cliff

    backandforth = elements((60.0, 6), atlas)
    assert len(backandforth) == 3
    assert isinstance(backandforth[0], Evolution)
    assert backandforth[0].cliff
    assert isinstance(backandforth[1], Matching)
    assert not backandforth[1].inverse

    down = elements((5.0, 3), atlas)
    assert all([isinstance(el, Evolution) for i, el in enumerate(down) if i % 2 == 0])
    assert all([isinstance(el, Matching) for i, el in enumerate(down) if i % 2 == 1])


def test_create(neweko: EKO):
    pass
