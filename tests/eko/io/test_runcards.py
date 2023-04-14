import copy
from dataclasses import fields
from io import StringIO

import numpy as np
import pytest
import yaml
from banana.data.theories import default_card as theory_card

from eko import interpolation
from eko.io import runcards as rc
from ekomark.data.operators import default_card as operator_card


class TestRotations:
    XGRID_TEST = [1e-3, 1e-2, 1e-1, 1.0]

    def test_serialization(self):
        rot = rc.Rotations(interpolation.XGrid(self.XGRID_TEST))

        d = rot.raw
        rot1 = rot.from_dict(d)

        for f in fields(rc.Rotations):
            assert getattr(rot, f.name) == getattr(rot1, f.name)

        assert d["targetgrid"] is None
        assert "_targetgrid" not in d

    def test_pids(self):
        rot = rc.Rotations(interpolation.XGrid(self.XGRID_TEST))

        # no check on correctness of value set
        rot.inputpids = [0, 1]
        # but the internal grid is unmodified
        assert len(rot.pids) == 14
        # and fallback implemented for unset external bases
        assert np.all(rot.targetpids == rot.pids)

    def test_grids(self):
        rot = rc.Rotations(interpolation.XGrid(self.XGRID_TEST))

        # no check on correctness of value set
        rot.inputgrid = interpolation.XGrid([0.1, 1])
        # but the internal grid is unmodified
        assert len(rot.xgrid) == len(self.XGRID_TEST)
        # and fallback implemented for unset external grids
        assert np.all(rot.targetgrid == rot.xgrid)


def test_flavored_mu2grid():
    mugrid = list(range(0, 40, 5))
    masses = [10, 20, 30]
    ratios = [1, 1, 1]

    flavored = rc.flavored_mugrid(mugrid, masses, ratios)
    assert pytest.approx([flav for _, flav in flavored]) == [3, 3, 4, 4, 5, 5, 6, 6]

    # check we can dump
    stream = StringIO()
    ser = yaml.safe_dump(flavored, stream)
    stream.seek(0)
    deser = yaml.safe_load(stream)
    assert len(flavored) == len(deser)
    # after deserialization on is now list instead of tuple
    for t, l in zip(flavored, deser):
        assert len(t) == len(l)
        assert t == tuple(l)


def test_runcards_ekomark():
    # check conversion
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operator_card)
    nt, no = rc.update(tc, oc)
    assert isinstance(nt, rc.TheoryCard)
    assert isinstance(no, rc.OperatorCard)
    # check is idempotent
    nnt, nno = rc.update(nt, no)
    assert nnt == nt
    assert nno == no
