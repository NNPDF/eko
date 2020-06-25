# -*- coding: utf-8 -*-
import io
import copy
from unittest import mock

import pytest
import numpy as np

from eko import output


class TestOutput:
    shape = (2, 2)

    def mkO(self):
        ma, mae = np.random.rand(2, *self.shape)
        return ma, mae

    def test_io(self):
        # build data
        xgrid = np.array([0.5, 1.0])
        polynomial_degree = 1
        is_log_interpolation = False
        q2_ref = 1
        VV, VVe = self.mkO()
        q2_out = 2
        Q2grid = {q2_out: {"operators": {"V.V": VV}, "operator_errors": {"V.V": VVe},}}
        d = dict(
            xgrid=xgrid,
            polynomial_degree=polynomial_degree,
            is_log_interpolation=is_log_interpolation,
            q2_ref=q2_ref,
            Q2grid=Q2grid,
        )
        # create object
        o1 = output.Output(d)
        # test streams
        stream = io.StringIO()
        o1.dump_yaml(stream)
        # rewind and read again
        stream.seek(0)
        o2 = output.Output.load_yaml(stream)
        np.testing.assert_almost_equal(o1["xgrid"], xgrid)
        np.testing.assert_almost_equal(o2["xgrid"], xgrid)
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
            np.testing.assert_almost_equal(o3["xgrid"], xgrid)

    def test_apply(self):
        # build data
        xgrid = np.array([0.5, 1.0])
        polynomial_degree = 1
        is_log_interpolation = False
        q2_ref = 1
        VV, VVe = self.mkO()
        q2_out = 2
        Q2grid = {q2_out: {"operators": {"V.V": VV}, "operator_errors": {"V.V": VVe},}}
        d = dict(
            xgrid=xgrid,
            polynomial_degree=polynomial_degree,
            is_log_interpolation=is_log_interpolation,
            q2_ref=q2_ref,
            Q2grid=Q2grid,
        )
        # create object
        o = output.Output(d)
        # fake pdfs
        V = lambda x: x
        V3 = lambda x: x
        pdf_grid = o.apply_pdf({"V": V, "V3": V3})
        assert len(pdf_grid) == 1
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == ["V"]
        np.testing.assert_almost_equal(pdfs["V"], VV @ V(xgrid))
        # rotate to target_grid
        target_grid = [0.75]
        pdf_grid = o.apply_pdf({"V": V, "V3": V3}, target_grid)
        assert len(pdf_grid) == 1
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == ["V"]
        # 0.75 is the the average of .5 and 1. -> mix equally
        np.testing.assert_almost_equal(pdfs["V"], (VV @ V(xgrid)) @ [0.5, 0.5])

    def test_get_op(self):
        # build data
        xgrid = np.array([0.5, 1.0])
        polynomial_degree = 1
        is_log_interpolation = False
        q2_ref = 1
        VV1, VVe1 = self.mkO()
        q2_out1 = 2
        VV2, VVe2 = self.mkO()
        q2_out2 = 3
        Q2grid = {
            q2_out1: {"operators": {"V.V": VV1}, "operator_errors": {"V.V": VVe1},},
            q2_out2: {"operators": {"V.V": VV2}, "operator_errors": {"V.V": VVe2},},
        }
        d = dict(
            xgrid=xgrid,
            polynomial_degree=polynomial_degree,
            is_log_interpolation=is_log_interpolation,
            q2_ref=q2_ref,
            Q2grid=Q2grid,
        )
        # create object
        o = output.Output(d)
        for q2 in [q2_out1, q2_out2]:
            ph = o.get_op(q2)
            raw = ph.get_raw_operators()
            np.testing.assert_almost_equal(
                raw["operators"]["V.V"], Q2grid[q2]["operators"]["V.V"]
            )
        # errors
        with pytest.raises(KeyError):
            o.get_op(q2_out2 + 1)

    def test_concat(self):
        # build data
        xgrid = np.array([0.5, 1.0])
        polynomial_degree = 1
        is_log_interpolation = False
        q2_ref = 1
        VVl, VVle = self.mkO()
        q2_out = 2
        Q2grid = {
            q2_out: {"operators": {"V.V": VVl}, "operator_errors": {"V.V": VVle},}
        }
        d = dict(
            xgrid=xgrid,
            polynomial_degree=polynomial_degree,
            is_log_interpolation=is_log_interpolation,
            q2_ref=q2_ref,
            Q2grid=Q2grid,
        )
        # create object
        o1 = output.Output(d)
        # prepare second
        d2 = copy.deepcopy(d)
        d2["q2_ref"] = q2_out
        VVh, VVhe = self.mkO()
        q2_final = 3
        d2["Q2grid"] = {
            q2_final: {"operators": {"V.V": VVh}, "operator_errors": {"V.V": VVhe},}
        }
        o2 = output.Output(d2)
        # join
        o21 = o2.concat(o1)
        assert isinstance(o21, output.Output)
        assert o21["q2_ref"] == q2_ref
        assert o21["is_log_interpolation"] == o1["is_log_interpolation"]
        assert q2_final in o21["Q2grid"]
        np.testing.assert_almost_equal(
            o21["Q2grid"][q2_final]["operators"]["V.V"], VVh @ VVl
        )

        # errors
        with pytest.raises(ValueError):
            o1.concat({})
        with pytest.raises(ValueError):
            dd = copy.deepcopy(d)
            dd["polynomial_degree"] += 1
            o1.concat(output.Output(dd))
        with pytest.raises(ValueError):
            dd = copy.deepcopy(d)
            dd["is_log_interpolation"] = not dd["is_log_interpolation"]
            o1.concat(output.Output(dd))
        with pytest.raises(ValueError):
            dd = copy.deepcopy(d)
            dd["xgrid"] = dd["xgrid"][:-1]
            o1.concat(output.Output(dd))
        with pytest.raises(ValueError):
            dd = copy.deepcopy(d)
            dd["xgrid"] = dd["xgrid"][:-1] + [dd["xgrid"][-1] * 0.9]
            o1.concat(output.Output(dd))
        # will fail due to non-matching scales
        with pytest.raises(ValueError):
            dd = copy.deepcopy(d)
            o1.concat(output.Output(dd))
