from dataclasses import fields

import numpy as np

from eko import interpolation
from eko.io import runcards as rc


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
