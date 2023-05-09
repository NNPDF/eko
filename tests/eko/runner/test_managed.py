import copy

import numpy as np

from eko import EKO
from eko.io.runcards import TheoryCard
from eko.runner.managed import solve

from . import check_shapes


def test_raw(theory_card, operator_card, tmp_path):
    """we don't check the content here, but only the shape"""
    path = tmp_path / "eko.tar"
    tc = theory_card
    oc = operator_card
    solve(tc, oc, path=path)
    with EKO.read(path) as eko_:
        check_shapes(eko_, eko_.xgrid, eko_.xgrid, tc, oc)


def test_vfns(theory_ffns, operator_card, tmp_path):
    path = tmp_path / "eko.tar"
    tc: TheoryCard = theory_ffns(3)
    oc = copy.deepcopy(operator_card)
    tc.heavy.matching_ratios.c = 1.0
    tc.heavy.matching_ratios.b = 1.0
    tc.order = (2, 0)
    solve(tc, oc, path=path)
    with EKO.read(path) as eko_:
        check_shapes(eko_, eko_.xgrid, eko_.xgrid, tc, oc)
