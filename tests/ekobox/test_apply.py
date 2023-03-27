import numpy as np

from ekobox import apply
from tests.conftest import EKOFactory


class TestApply:
    def test_apply(self, eko_factory: EKOFactory, fake_pdf):
        eko_factory.operator.rotations.pids = np.array([0, 1])
        eko = eko_factory.get()
        mu2_out = eko.mu2grid[0]
        # fake pdfs
        pdf_grid = apply.apply_pdf(eko, fake_pdf)
        assert len(pdf_grid) == len(eko.mu2grid)
        pdfs = pdf_grid[mu2_out]["pdfs"]
        assert list(pdfs.keys()) == list(eko.rotations.targetpids)
        ref_pid1 = eko[mu2_out].operator[0, :, 1, :] @ np.ones(len(eko.xgrid))
        np.testing.assert_allclose(pdfs[0], ref_pid1)
        ref_pid2 = eko[mu2_out].operator[1, :, 1, :] @ np.ones(len(eko.xgrid))
        np.testing.assert_allclose(pdfs[1], ref_pid2)
        # rotate to target_grid
        target_grid = [0.75]
        pdf_grid = apply.apply_pdf(eko, fake_pdf, target_grid)
        assert len(pdf_grid) == 1
        pdfs = pdf_grid[mu2_out]["pdfs"]
        assert list(pdfs.keys()) == list(eko.rotations.targetpids)

    def test_apply_flavor(self, eko_factory: EKOFactory, fake_pdf, monkeypatch):
        eko_factory.operator.rotations.pids = np.array([0, 1])
        eko = eko_factory.get()
        q2_out = eko.mu2grid[0]
        # fake pdfs
        monkeypatch.setattr(
            "eko.basis_rotation.rotate_flavor_to_evolution", np.ones((2, 2))
        )
        monkeypatch.setattr(
            "eko.basis_rotation.flavor_basis_pids",
            eko.rotations.targetpids,
        )
        fake_evol_basis = ("a", "b")
        monkeypatch.setattr("eko.basis_rotation.evol_basis", fake_evol_basis)
        pdf_grid = apply.apply_pdf(eko, fake_pdf, rotate_to_evolution_basis=True)
        assert len(pdf_grid) == len(eko.mu2grid)
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == list(fake_evol_basis)
        ref_a = (
            eko[q2_out].operator[0, :, 1, :] + eko[q2_out].operator[1, :, 1, :]
        ) @ np.ones(len(eko.rotations.xgrid))
        np.testing.assert_allclose(pdfs["a"], ref_a)
