import copy

import numpy as np
import pytest
from banana.data.theories import default_card as theory_card

from eko.io import runcards as rc
from ekomark.data import update_runcards
from ekomark.data.operators import default_card as operator_card

from ...eko.io.test_runcards import check_dumpable


def test_runcards_opcard():
    # check conversion
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operator_card)
    tc["Q0"] = 2.0
    # mugrid
    oc["mugrid"] = [2.0, 10.0]
    _nt, no = update_runcards(tc, oc)
    assert isinstance(no, rc.OperatorCard)
    assert len(no.evolgrid) == len(oc["mugrid"])
    assert len(no.mu2grid) == len(no.evolgrid)
    assert no.evolgrid[0][-1] == 4
    assert no.evolgrid[1][-1] == 5
    np.testing.assert_allclose(no.init[0], tc["Q0"])
    np.testing.assert_allclose(no.mu20, tc["Q0"] ** 2.0)
    assert len(no.pids) == 14
    check_dumpable(no)
    del oc["mugrid"]
    # or mu2grid
    oc["mu2grid"] = [9.0, 30.0, 32.0]
    _nt, no = update_runcards(tc, oc)
    assert isinstance(no, rc.OperatorCard)
    assert len(no.evolgrid) == len(oc["mu2grid"])
    assert len(no.mu2grid) == len(no.evolgrid)
    assert no.evolgrid[0][-1] == 4
    assert no.evolgrid[1][-1] == 5
    assert no.evolgrid[2][-1] == 5
    check_dumpable(no)
    del oc["mu2grid"]
    # or Q2grid
    oc["Q2grid"] = [15.0, 130.0, 140.0, 1e5]
    _nt, no = update_runcards(tc, oc)
    assert isinstance(no, rc.OperatorCard)
    assert len(no.evolgrid) == len(oc["Q2grid"])
    assert len(no.mu2grid) == len(no.evolgrid)
    assert no.evolgrid[0][-1] == 4
    assert no.evolgrid[1][-1] == 5
    assert no.evolgrid[2][-1] == 5
    assert no.evolgrid[3][-1] == 6
    check_dumpable(no)


def test_runcards_ekomark():
    # check conversion
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operator_card)
    nt, no = update_runcards(tc, oc)
    assert isinstance(nt, rc.TheoryCard)
    assert isinstance(no, rc.OperatorCard)
    # check is idempotent
    nnt, nno = update_runcards(nt, no)
    assert nnt == nt
    assert nno == no


def test_runcards_quarkmass():
    tc = copy.deepcopy(theory_card)
    tc["nfref"] = 5
    tc["IC"] = 1
    oc = copy.deepcopy(operator_card)
    nt, no = update_runcards(tc, oc)
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
    nt, no = update_runcards(tc, oc)
    for _, scale in nt.heavy.masses:
        assert not np.isnan(scale)
    m2s = rc.masses(nt, no.configs.evolution_method)
    for m1, m2 in zip(m2s, raw2):
        assert not np.isclose(m1, m2)
    tc["HQ"] = "Blub"
    with pytest.raises(ValueError, match="mass scheme"):
        _nt, _no = update_runcards(tc, oc)
    nt.heavy.masses_scheme = "Bla"
    with pytest.raises(ValueError, match="mass scheme"):
        _ms = rc.masses(nt, no.configs.evolution_method)
