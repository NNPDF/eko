import copy
import io
import pathlib
import shutil
import tempfile

import numpy as np
import pytest

from eko import output
from eko.output import legacy


class TestLegacy:
    def test_io(self, fake_legacy, tmp_path):
        # create object
        o1, fake_card = fake_legacy
        for q2, op in fake_card["Q2grid"].items():
            o1[q2] = output.Operator.from_dict(op)

        # test streams
        stream = io.StringIO()
        legacy.dump_yaml(o1, stream)
        # rewind and read again
        stream.seek(0)
        o2 = legacy.load_yaml(stream)
        np.testing.assert_almost_equal(o1.xgrid.raw, fake_card["interpolation_xgrid"])
        np.testing.assert_almost_equal(o2.xgrid.raw, fake_card["interpolation_xgrid"])
        # fake output files
        fpyaml = tmp_path / "test.yaml"
        legacy.dump_yaml_to_file(o1, fpyaml)
        # fake input file
        o3 = legacy.load_yaml_from_file(fpyaml)
        np.testing.assert_almost_equal(o3.xgrid.raw, fake_card["interpolation_xgrid"])
        # repeat for tar
        fptar = tmp_path / "test.tar"
        legacy.dump_tar(o1, fptar)
        o4 = legacy.load_tar(fptar)
        np.testing.assert_almost_equal(o4.xgrid.raw, fake_card["interpolation_xgrid"])
        fn = "test"
        with pytest.raises(ValueError, match="wrong suffix"):
            legacy.dump_tar(o1, fn)

    def test_tocards(self, fake_legacy):
        o1, _fake_card = fake_legacy
        raw = legacy.get_raw(o1)
        card = legacy.tocard(raw)
        assert "Q0" in card
        rraw = copy.deepcopy(raw)
        del rraw["Q0"]
        ccard = legacy.tocard(rraw)
        assert "Q0" in ccard

    def test_rename_issue81(self, fake_legacy):
        # https://github.com/N3PDF/eko/issues/81
        # create object
        o1, fake_card = fake_legacy
        for q2, op in fake_card["Q2grid"].items():
            o1[q2] = output.Operator.from_dict(op)

        with tempfile.TemporaryDirectory() as folder:
            # dump
            p = pathlib.Path(folder)
            fp1 = p / "test1.tar"
            fp2 = p / "test2.tar"
            legacy.dump_tar(o1, fp1)
            # rename
            shutil.move(fp1, fp2)
            # reload
            o4 = legacy.load_tar(fp2)
            np.testing.assert_almost_equal(
                o4.xgrid.raw, fake_card["interpolation_xgrid"]
            )

    def test_io_bin(self, fake_legacy):
        # create object
        o1, fake_card = fake_legacy
        for q2, op in fake_card["Q2grid"].items():
            o1[q2] = output.Operator.from_dict(op)
        # test streams
        stream = io.StringIO()
        legacy.dump_yaml(o1, stream, False)
        # rewind and read again
        stream.seek(0)
        o2 = legacy.load_yaml(stream)
        np.testing.assert_almost_equal(o1.xgrid.raw, fake_card["interpolation_xgrid"])
        np.testing.assert_almost_equal(o2.xgrid.raw, fake_card["interpolation_xgrid"])
