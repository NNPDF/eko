from pathlib import Path

import numpy as np
import pytest

from eko import EKO
from eko.io.items import Operator
from eko.io.runcards import OperatorCard, TheoryCard
from eko.runner import commons, recipes


@pytest.fixture
def neweko(theory_card: TheoryCard, operator_card: OperatorCard, tmp_path: Path):
    path = tmp_path / "eko.tar"
    with EKO.create(path) as builder:
        yield builder.load_cards(theory_card, operator_card).build()

    path.unlink(missing_ok=True)


@pytest.fixture
def identity(neweko: EKO):
    xs = len(neweko.xgrid.raw)
    flavs = len(neweko.bases.pids)
    return Operator(operator=np.eye(xs * flavs).reshape((xs, flavs, xs, flavs)))


@pytest.fixture
def ekoparts(neweko: EKO, identity: Operator):
    atlas = commons.atlas(neweko.theory_card, neweko.operator_card)
    neweko.load_recipes(recipes.create(neweko.operator_card.evolgrid, atlas))

    for rec in neweko.recipes:
        neweko.parts[rec] = identity
    for rec in neweko.recipes_matching:
        neweko.parts_matching[rec] = identity

    return neweko
