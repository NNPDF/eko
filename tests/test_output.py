# -*- coding: utf-8 -*-
import io
import copy
from unittest import mock

import pytest
import numpy as np

from eko import output


class FakePDF:
    def hasFlavor(self, *_args):
        return True

    def xfxQ2(self, _pid, x, _q2):
        return x


class TestOutput:
    shape = (2, 2)

    def mkO(self):
        ma, mae = np.random.rand(2, *self.shape)
        return ma, mae

    def mk_g(self, q2s, lpids, lx):
        Q2grid = {}
        for q2 in q2s:
            Q2grid[q2] = {
                "operators": np.random.rand(lpids, lx, lpids, lx),
                "operator_errors": np.random.rand(lpids, lx, lpids, lx),
            }
        return Q2grid

    def fake_output(self):
        # build data
        interpolation_xgrid = np.array([0.5, 1.0])
        interpolation_polynomial_degree = 1
        interpolation_is_log = False
        pids = [1, 2]
        q2_ref = 1
        q2_out = 2
        Q2grid = self.mk_g([q2_out], len(pids), len(interpolation_xgrid))
        d = dict(
            interpolation_xgrid=interpolation_xgrid,
            interpolation_polynomial_degree=interpolation_polynomial_degree,
            interpolation_is_log=interpolation_is_log,
            q2_ref=q2_ref,
            pids=pids,
            Q2grid=Q2grid,
        )
        return d

    def test_io(self):
        d = self.fake_output()
        # create object
        o1 = output.Output(d)
        # test streams
        stream = io.StringIO()
        o1.dump_yaml(stream)
        # rewind and read again
        stream.seek(0)
        o2 = output.Output.load_yaml(stream)
        np.testing.assert_almost_equal(
            o1["interpolation_xgrid"], d["interpolation_xgrid"]
        )
        np.testing.assert_almost_equal(
            o2["interpolation_xgrid"], d["interpolation_xgrid"]
        )
        # fake output files
        m_out = mock.mock_open(read_data="")
        with mock.patch("builtins.open", m_out) as mock_file:
            fn = "test.yaml"
            o1.dump_yaml_to_file(fn)
            mock_file.assert_called_with(fn, "w")
        # fake input file
        stream.seek(0)
        m_in = mock.mock_open(read_data=stream.getvalue())
        with mock.patch("builtins.open", m_in) as mock_file:
            fn = "test.yaml"
            o3 = output.Output.load_yaml_from_file(fn)
            mock_file.assert_called_with(fn)
            np.testing.assert_almost_equal(
                o3["interpolation_xgrid"], d["interpolation_xgrid"]
            )

    def test_apply(self):
        d = self.fake_output()
        q2_out = list(d["Q2grid"].keys())[0]
        # create object
        o = output.Output(d)
        # fake pdfs
        pdf = FakePDF()
        pdf_grid = o.apply_pdf(pdf)
        assert len(pdf_grid) == 1
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == d["pids"]
        # np.testing.assert_almost_equal(pdfs[1], pdfs[-1])
        # rotate to target_grid
        target_grid = [0.75]
        pdf_grid = o.apply_pdf(pdf, target_grid)
        assert len(pdf_grid) == 1
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == d["pids"]
        # np.testing.assert_almost_equal(pdfs[1], pdfs[-1])
        # 0.75 is the the average of .5 and 1. -> mix equally
        # np.testing.assert_almost_equal(
        #    pdfs["V"], (VV @ interpolation_xgrid) @ [0.5, 0.5]
        # )
