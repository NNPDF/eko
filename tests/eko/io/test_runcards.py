import copy
from io import StringIO

import numpy as np
import pytest
import yaml
from banana.data.theories import default_card as theory_card

from eko import interpolation
from eko.io import runcards as rc
from eko.io.bases import Bases
from ekomark.data.operators import default_card as operator_card


def test_flavored_mu2grid():
    mugrid = list(range(5, 40, 5))
    masses = [10, 20, 30]
    ratios = [1, 1, 1]

    flavored = rc.flavored_mugrid(mugrid, masses, ratios)
    assert pytest.approx([flav for _, flav in flavored]) == [3, 4, 4, 5, 5, 6, 6]

    # check we can dump
    stream = StringIO()
    _ = yaml.safe_dump(flavored, stream)
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
