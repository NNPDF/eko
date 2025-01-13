import numpy as np

from eko import basis_rotation as br
from ekobox import apply
from tests.conftest import EKOFactory


def test_apply(eko_factory: EKOFactory, fake_pdf):
    eko = eko_factory.get()
    ep_out = eko.evolgrid[0]
    # base application
    pdfs, _errors = apply.apply_pdf(eko, fake_pdf)
    assert len(pdfs) == len(eko.evolgrid)
    ep_pdfs = pdfs[ep_out]
    assert list(ep_pdfs.keys()) == list(br.flavor_basis_pids)
    # rotate to target_grid
    for target_grid in ([0.75], [0.5, 0.6, 0.7]):
        pdfs, _errors = apply.apply_pdf(eko, fake_pdf, target_grid)
        ep_pdfs = pdfs[ep_out]
        assert list(ep_pdfs.keys()) == list(br.flavor_basis_pids)
        assert len(ep_pdfs[21]) == len(target_grid)
    # rotate flavor
    pdfs, _errors = apply.apply_pdf(eko, fake_pdf, rotate_to_evolution_basis=True)
    ep_pdfs = pdfs[ep_out]
    assert list(ep_pdfs.keys()) == list(br.evol_basis_pids)


def test_apply_grids(eko_factory: EKOFactory):
    eko = eko_factory.get()
    # since everything is random, we can only test the tensor shapes here
    input_grids = np.random.rand(3, len(br.flavor_basis_pids), len(eko.xgrid))
    pdfs, _errors = apply.apply_grids(eko, input_grids)
    for _ep, res in pdfs.items():
        assert res.shape == input_grids.shape
