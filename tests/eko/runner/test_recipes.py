from pathlib import Path

import pytest

from eko import EKO
from eko.io.runcards import OperatorCard, TheoryCard


@pytest.fixture
def neweko(theory_card: TheoryCard, operator_card: OperatorCard, tmp_path: Path):
    path = tmp_path / "eko.tar"
    with EKO.create(path) as builder:
        yield builder.load_cards(theory_card, operator_card).build()

    path.unlink(missing_ok=True)


def test_create(neweko: EKO):
    pass
