import copy
import enum

import numpy as np
import pytest

import eko
from eko import EKO
from eko.quantities.heavy_quarks import QuarkMassScheme


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
    theory_card.quark_masses_scheme = FakeEM.BLUB
    with pytest.raises(ValueError, match="BLUB"):
        eko.runner.legacy.Runner(theory_card, operator_card, path=path)
    # MSbar scheme
    theory_card.quark_masses_scheme = QuarkMassScheme.MSBAR
    theory_card.couplings.num_flavs_ref = 5
    theory_card.quark_masses.c.scale = 2
    theory_card.quark_masses.b.scale = 4.5
    theory_card.quark_masses.t.scale = 173.07
    r = eko.runner.legacy.Runner(theory_card, operator_card, path=path)
    r.compute()
    with EKO.read(path) as eko_:
        check_shapes(eko_, eko_.xgrid, eko_.xgrid, theory_card, operator_card)


def check_shapes(o, txs, ixs, theory_card, operators_card):
    tpids = len(o.rotations.targetpids)
    ipids = len(o.rotations.inputpids)
    op_shape = (tpids, len(txs), ipids, len(ixs))

    # check output = input
    np.testing.assert_allclose(o.xgrid.raw, operators_card.rotations.xgrid.raw)
    # targetgrid and inputgrid in the opcard are now ignored, we are testing this
    np.testing.assert_allclose(
        o.rotations.targetgrid.raw,
        txs.raw,
    )
    np.testing.assert_allclose(o.rotations.inputgrid.raw, ixs.raw)
    np.testing.assert_allclose(o.mu20, operators_card.mu20)
    # check available operators
    assert len(o.mu2grid) == len(operators_card.mu2grid)
    assert list(o.mu2grid) == operators_card.mu2grid
    for _, ops in o.items():
        assert ops.operator.shape == op_shape
        assert ops.error.shape == op_shape


def test_vfns(theory_ffns, operator_card, tmp_path):
    # change targetpids
    path = tmp_path / "eko.tar"
    tc = theory_ffns(3)
    oc = copy.deepcopy(operator_card)
    tc.matching.c = 1.0
    tc.matching.b = 1.0
    tc.order = (2, 0)
    oc.debug.skip_non_singlet = False
    r = eko.runner.legacy.Runner(tc, oc, path=path)
    r.compute()
    with EKO.read(path) as eko_:
        check_shapes(eko_, eko_.xgrid, eko_.xgrid, tc, oc)
