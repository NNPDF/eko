from dataclasses import fields

import numpy as np

from eko import basis_rotation as br
from eko import interpolation
from eko.io.bases import Bases


class TestBases:
    XGRID_TEST = [1e-3, 1e-2, 1e-1, 1.0]

    def test_serialization(self):
        rot = Bases(interpolation.XGrid(self.XGRID_TEST))

        d = rot.raw
        rot1 = rot.from_dict(d)

        for f in fields(Bases):
            assert getattr(rot, f.name) == getattr(rot1, f.name)

        assert d["targetgrid"] is None
        assert "_targetgrid" not in d

    def test_pids(self):
        rot = Bases(interpolation.XGrid(self.XGRID_TEST))

        # no check on correctness of value set
        rot.inputpids = [0, 1]
        # but the internal grid is unmodified
        assert len(rot.pids) == 14
        # and fallback implemented for unset external bases
        assert np.all(rot.targetpids == rot.pids)

    def test_grids(self):
        rot = Bases(interpolation.XGrid(self.XGRID_TEST))

        # no check on correctness of value set
        rot.inputgrid = interpolation.XGrid([0.1, 1])
        # but the internal grid is unmodified
        assert len(rot.xgrid) == len(self.XGRID_TEST)
        # and fallback implemented for unset external grids
        assert np.all(rot.targetgrid == rot.xgrid)

    def test_fallback(self):
        xg = interpolation.XGrid([0.1, 1.0])
        r = Bases(xgrid=xg)
        np.testing.assert_allclose(r.targetpids, r.pids)
        np.testing.assert_allclose(r.inputpids, r.pids)
        assert r.xgrid == xg
        assert r.targetgrid == xg
        assert r.inputgrid == xg

    def test_overwrite(self):
        tpids = np.array([3, 4] + list(br.flavor_basis_pids[2:]))
        ipids = np.array([5, 6] + list(br.flavor_basis_pids[2:]))
        xg = interpolation.XGrid([0.1, 1.0])
        txg = interpolation.XGrid([0.2, 1.0])
        ixg = interpolation.XGrid([0.3, 1.0])
        r = Bases(
            xgrid=xg,
            _targetgrid=txg,
            _inputgrid=ixg,
            _targetpids=tpids,
            _inputpids=ipids,
        )
        np.testing.assert_allclose(r.targetpids, tpids)
        np.testing.assert_allclose(r.inputpids, ipids)
        assert r.xgrid == xg
        assert r.targetgrid == txg
        assert r.inputgrid == ixg

    def test_init(self):
        xg = interpolation.XGrid([0.1, 1.0])
        txg = np.array([0.2, 1.0])
        ixg = {"grid": [0.3, 1.0], "log": True}
        r = Bases(xgrid=xg, _targetgrid=txg, _inputgrid=ixg)
        assert isinstance(r.xgrid, interpolation.XGrid)
        assert isinstance(r.targetgrid, interpolation.XGrid)
        assert isinstance(r.inputgrid, interpolation.XGrid)
        assert r.xgrid == xg
        assert r.targetgrid == interpolation.XGrid(txg)
        assert r.inputgrid == interpolation.XGrid.load(ixg)
