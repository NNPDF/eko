import numpy as np

from ekobox import apply
from tests.conftest import EKOFactory


class TestApply:
    def test_apply(self, eko_factory: EKOFactory, fake_pdf):
        eko = eko_factory.get()
        ep_out = eko.evolgrid[0]
        # fake pdfs
        pdf_grid = apply.apply_pdf(eko, fake_pdf)
        assert len(pdf_grid) == len(eko.evolgrid)
        pdfs = pdf_grid[ep_out]["pdfs"]
        assert list(pdfs.keys()) == list(eko.rotations.targetpids)
        # rotate to target_grid
        target_grid = [0.75]
        pdf_grid = apply.apply_pdf(eko, fake_pdf, target_grid)
        assert len(pdf_grid) == 1
        pdfs = pdf_grid[ep_out]["pdfs"]
        assert list(pdfs.keys()) == list(eko.rotations.targetpids)

    def test_apply_flavor(self, eko_factory: EKOFactory, fake_pdf, monkeypatch):
        eko = eko_factory.get()
        ep_out = eko.evolgrid[0]
        # fake pdfs
        monkeypatch.setattr(
            "eko.basis_rotation.rotate_flavor_to_evolution", np.ones((14, 14))
        )
        monkeypatch.setattr(
            "eko.basis_rotation.flavor_basis_pids",
            eko.rotations.targetpids,
        )
        fake_evol_basis = tuple(
            ["a", "b"] + [str(x) for x in range(len(eko.rotations.pids) - 2)]
        )
        monkeypatch.setattr("eko.basis_rotation.evol_basis", fake_evol_basis)
        pdf_grid = apply.apply_pdf(eko, fake_pdf, rotate_to_evolution_basis=True)
        assert len(pdf_grid) == len(eko.evolgrid)
        pdfs = pdf_grid[ep_out]["pdfs"]
        assert list(pdfs.keys()) == list(fake_evol_basis)
