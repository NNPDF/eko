from typing import List

from eko.io.items import Evolution, Matching
from eko.io.types import EvolutionPoint
from eko.matchings import Atlas
from eko.quantities.heavy_quarks import MatchingScales
from eko.runner.recipes import create, elements

SCALES = MatchingScales([10.0, 20.0, 30.0])
ATLAS = Atlas(SCALES, (50.0, 5))


def test_elements():
    onestep = elements((60.0, 5), ATLAS)
    assert len(onestep) == 1
    assert isinstance(onestep[0], Evolution)
    assert not onestep[0].cliff

    backandforth = elements((60.0, 6), ATLAS)
    assert len(backandforth) == 3
    assert isinstance(backandforth[0], Evolution)
    assert backandforth[0].cliff
    assert isinstance(backandforth[1], Matching)
    assert not backandforth[1].inverse

    down = elements((5.0, 3), ATLAS)
    assert all([isinstance(el, Evolution) for i, el in enumerate(down) if i % 2 == 0])
    assert all([isinstance(el, Matching) for i, el in enumerate(down) if i % 2 == 1])


def test_create():
    evolgrid: List[EvolutionPoint] = [(60.0, 5)]

    recs = create(evolgrid, ATLAS)
    assert len(recs) == 1

    evolgrid.append((60.0, 6))
    recs = create(evolgrid, ATLAS)
    assert len(recs) == 1 + 3

    evolgrid.append((70.0, 6))
    recs = create(evolgrid, ATLAS)
    assert len(recs) == 1 + 3 + 1
