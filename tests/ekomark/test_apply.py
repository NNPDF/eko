# -*- coding: utf-8 -*-
import numpy as np

from ekomark import apply


class TestApply:
    def test_apply(self, fake_legacy, fake_pdf):
        o, fake_card = fake_legacy
        q2_out = list(fake_card["Q2grid"].keys())[0]
        # fake pdfs
        pdf_grid = apply.apply_pdf(o, fake_pdf)
        assert len(pdf_grid) == len(fake_card["Q2grid"])
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == o.rotations.targetpids
        ref_pid1 = fake_card["Q2grid"][q2_out]["operator"][0, :, 1, :] @ np.ones(
            len(o.rotations.xgrid)
        )
        np.testing.assert_allclose(pdfs[0], ref_pid1)
        ref_pid2 = fake_card["Q2grid"][q2_out]["operator"][1, :, 1, :] @ np.ones(
            len(o.rotations.xgrid)
        )
        np.testing.assert_allclose(pdfs[1], ref_pid2)
        # rotate to target_grid
        target_grid = [0.75]
        pdf_grid = apply.apply_pdf(o, fake_pdf, target_grid)
        assert len(pdf_grid) == 1
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == o.rotations.targetpids

    def test_apply_flavor(self, fake_legacy, fake_pdf, monkeypatch):
        o, fake_card = fake_legacy
        q2_out = list(fake_card["Q2grid"].keys())[0]
        # fake pdfs
        monkeypatch.setattr(
            "eko.basis_rotation.rotate_flavor_to_evolution", np.ones((2, 2))
        )
        monkeypatch.setattr(
            "eko.basis_rotation.flavor_basis_pids",
            o.rotations.targetpids,
        )
        fake_evol_basis = ("a", "b")
        monkeypatch.setattr("eko.basis_rotation.evol_basis", fake_evol_basis)
        pdf_grid = apply.apply_pdf(o, fake_pdf, rotate_to_evolution_basis=True)
        assert len(pdf_grid) == len(fake_card["Q2grid"])
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == list(fake_evol_basis)
        ref_a = (
            fake_card["Q2grid"][q2_out]["operator"][0, :, 1, :]
            + fake_card["Q2grid"][q2_out]["operator"][1, :, 1, :]
        ) @ np.ones(len(o.rotations.xgrid))
        np.testing.assert_allclose(pdfs["a"], ref_a)
