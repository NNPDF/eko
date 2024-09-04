import copy
import enum

import numpy as np
import pytest

import eko
from eko import EKO
from eko.io.runcards import TheoryCard
from eko.quantities.heavy_quarks import QuarkMassScheme

from . import check_shapes


def test_raw(theory_card, operator_card, tmp_path):
    """we don't check the content here, but only the shape"""
    path = tmp_path / "eko.tar"
    tc = theory_card
    oc = operator_card
    r = eko.runner.legacy.Runner(tc, oc, path=path)
    r.compute()
    with EKO.read(path) as eko_:
        check_shapes(eko_, eko_.xgrid, eko_.xgrid, tc, oc)


def test_mass_scheme(theory_card, operator_card, tmp_path):
    """we don't check the content here, but only the shape"""

    # wrong mass scheme
    class FakeEM(enum.Enum):
        BLUB = "blub"

    path = tmp_path / "eko.tar"
    theory_card.heavy.masses_scheme = FakeEM.BLUB
    with pytest.raises(ValueError, match="BLUB"):
        eko.runner.legacy.Runner(theory_card, operator_card, path=path)
    # MSbar scheme
    theory_card.heavy.masses_scheme = QuarkMassScheme.MSBAR
    theory_card.couplings.ref = (91.0, 5)
    theory_card.heavy.masses.c.scale = 2
    theory_card.heavy.masses.b.scale = 4.5
    theory_card.heavy.masses.t.scale = 173.07
    r = eko.runner.legacy.Runner(theory_card, operator_card, path=path)
    r.compute()
    with EKO.read(path) as eko_:
        check_shapes(eko_, eko_.xgrid, eko_.xgrid, theory_card, operator_card)


def test_vfns(theory_ffns, operator_card, tmp_path):
    path = tmp_path / "eko.tar"
    tc: TheoryCard = theory_ffns(3)
    oc = copy.deepcopy(operator_card)
    tc.heavy.matching_ratios.c = 1.0
    tc.heavy.matching_ratios.b = 1.0
    tc.order = (2, 0)
    r = eko.runner.legacy.Runner(tc, oc, path=path)
    r.compute()
    with EKO.read(path) as eko_:
        check_shapes(eko_, eko_.xgrid, eko_.xgrid, tc, oc)
