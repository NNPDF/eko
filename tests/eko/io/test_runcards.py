import copy
from io import StringIO

import numpy as np
import pytest
import yaml
from banana.data.theories import default_card as theory_card

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
    # after deserialization one is now list instead of tuple
    for t, l in zip(flavored, deser):
        assert len(t) == len(l)
        assert t == tuple(l)


def test_runcards_opcard():
    # check conversion
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operator_card)
    tc["Q0"] = 2.0
    # mugrid
    oc["mugrid"] = [2.0, 10.0]
    _nt, no = rc.update(tc, oc)
    assert isinstance(no, rc.OperatorCard)
    assert len(no.evolgrid) == len(oc["mugrid"])
    assert len(no.mu2grid) == len(no.evolgrid)
    assert no.evolgrid[0][-1] == 4
    assert no.evolgrid[1][-1] == 5
    np.testing.assert_allclose(no.mu0, tc["Q0"])
    np.testing.assert_allclose(no.mu20, tc["Q0"] ** 2.0)
    assert len(no.pids) == 14
    del oc["mugrid"]
    # or mu2grid
    oc["mu2grid"] = [9.0, 30.0, 32.0]
    _nt, no = rc.update(tc, oc)
    assert isinstance(no, rc.OperatorCard)
    assert len(no.evolgrid) == len(oc["mu2grid"])
    assert len(no.mu2grid) == len(no.evolgrid)
    assert no.evolgrid[0][-1] == 4
    assert no.evolgrid[1][-1] == 5
    assert no.evolgrid[2][-1] == 5
    del oc["mu2grid"]
    # or Q2grid
    oc["Q2grid"] = [15.0, 130.0, 140.0, 1e5]
    _nt, no = rc.update(tc, oc)
    assert isinstance(no, rc.OperatorCard)
    assert len(no.evolgrid) == len(oc["Q2grid"])
    assert len(no.mu2grid) == len(no.evolgrid)
    assert no.evolgrid[0][-1] == 4
    assert no.evolgrid[1][-1] == 5
    assert no.evolgrid[2][-1] == 5
    assert no.evolgrid[3][-1] == 6


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


def test_runcards_quarkmass():
    tc = copy.deepcopy(theory_card)
    tc["nfref"] = 5
    tc["IC"] = 1
    oc = copy.deepcopy(operator_card)
    nt, no = rc.update(tc, oc)
    assert nt.heavy.intrinsic_flavors == [4]
    for _, scale in nt.heavy.masses:
        assert np.isnan(scale)
    m2s = rc.masses(nt, no.configs.evolution_method)
    raw = rc.Legacy.heavies("m%s", tc)
    raw2 = np.power(raw, 2.0)
    np.testing.assert_allclose(m2s, raw2)
    tc["HQ"] = "MSBAR"
    tc["Qmc"] = raw[0] * 1.1
    tc["Qmb"] = raw[1] * 1.1
    tc["Qmt"] = raw[2] * 0.9
    nt, no = rc.update(tc, oc)
    for _, scale in nt.heavy.masses:
        assert not np.isnan(scale)
    m2s = rc.masses(nt, no.configs.evolution_method)
    for m1, m2 in zip(m2s, raw2):
        assert not np.isclose(m1, m2)
    tc["HQ"] = "Blub"
    with pytest.raises(ValueError, match="mass scheme"):
        _nt, _no = rc.update(tc, oc)
    nt.heavy.masses_scheme = "Bla"
    with pytest.raises(ValueError, match="mass scheme"):
        _ms = rc.masses(nt, no.configs.evolution_method)


def test_legacy_fallback():
    assert rc.Legacy.fallback(1, 2, 3) == 1
    assert rc.Legacy.fallback(None, 2, 3) == 2
    assert rc.Legacy.fallback(None, 2, 3, default=7) == 2
    assert rc.Legacy.fallback(None, None, 3) == 3
    assert rc.Legacy.fallback(None, None, None) is None
    assert rc.Legacy.fallback(None, None, None, default=7) == 7
