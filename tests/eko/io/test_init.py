import pathlib

import pytest

from eko import EKO
from eko.io import legacy


class TestLegacy:
    def test_load_tar(self, out_v0, tmp_path: pathlib.Path):
        oppath = tmp_path / "eko.tar"
        with pytest.warns(UserWarning, match="alphas"):
            legacy.load_tar(out_v0, oppath)
        with EKO.read(oppath) as eko:
            assert eko.metadata.data_version == 0
