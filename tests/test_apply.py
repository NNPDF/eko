# -*- coding: utf-8 -*-
import copy
import io
import pathlib
import shutil
import tempfile
from unittest import mock

import numpy as np
import pytest

from eko import basis_rotation as br
from eko import output
from ekomark import apply

from .test_output import FakeOutput, FakePDF


class TestApply(FakeOutput):
    def test_apply(self):
        d = self.fake_output()
        q2_out = list(d["Q2grid"].keys())[0]
        # create object
        o = output.Output(d)
        # fake pdfs
        pdf = FakePDF()
        pdf_grid = apply.apply_pdf(o, pdf)
        assert len(pdf_grid) == len(d["Q2grid"])
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == d["targetpids"]
        ref_pid1 = d["Q2grid"][q2_out]["operators"][0, :, 1, :] @ np.ones(
            len(d["interpolation_xgrid"])
        )
        np.testing.assert_allclose(pdfs[0], ref_pid1)
        ref_pid2 = d["Q2grid"][q2_out]["operators"][1, :, 1, :] @ np.ones(
            len(d["interpolation_xgrid"])
        )
        np.testing.assert_allclose(pdfs[1], ref_pid2)
        # rotate to target_grid
        target_grid = [0.75]
        pdf_grid = apply.apply_pdf(o, pdf, target_grid)
        assert len(pdf_grid) == 1
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == d["targetpids"]

    def test_apply_flavor(self, monkeypatch):
        d = self.fake_output()
        q2_out = list(d["Q2grid"].keys())[0]
        # create object
        o = output.Output(d)
        # fake pdfs
        pdf = FakePDF()
        monkeypatch.setattr(
            "eko.basis_rotation.rotate_flavor_to_evolution", np.ones((2, 2))
        )
        monkeypatch.setattr("eko.basis_rotation.flavor_basis_pids", d["targetpids"])
        fake_evol_basis = ("a", "b")
        monkeypatch.setattr("eko.basis_rotation.evol_basis", fake_evol_basis)
        pdf_grid = apply.apply_pdf(o, pdf, rotate_to_evolution_basis=True)
        assert len(pdf_grid) == len(d["Q2grid"])
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == list(fake_evol_basis)
        ref_a = (
            d["Q2grid"][q2_out]["operators"][0, :, 1, :]
            + d["Q2grid"][q2_out]["operators"][1, :, 1, :]
        ) @ np.ones(len(d["interpolation_xgrid"]))
        np.testing.assert_allclose(pdfs["a"], ref_a)
