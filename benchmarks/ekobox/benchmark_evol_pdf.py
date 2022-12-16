import pathlib

import numpy as np
import pytest

from eko import basis_rotation as br
from eko.interpolation import XGrid
from eko.io.runcards import OperatorCard, TheoryCard
from ekobox import evol_pdf as ev_p
from ekobox.genpdf import load

test_pdf = pathlib.Path(__file__).parent / "fakepdf"
lhapdf = pytest.importorskip("lhapdf")


@pytest.mark.isolated
def benchmark_evolve_single_member(
    tmp_path, cd, lhapdf_path, theory_card: TheoryCard, operator_card: OperatorCard
):
    mu2grid = [20.0, 100.0, 10000.0]
    theory = theory_card
    theory.order = (1, 0)
    theory.couplings.alphas.value = 0.118000
    theory.couplings.alphas.scale = 91.1876
    theory.couplings.alphaem.value = 0.007496
    theory.num_flavs_max_as = 3
    theory.num_flavs_max_pdf = 3
    theory.quark_masses.c.value = 1.3
    theory.quark_masses.b.value = 4.75
    theory.quark_masses.t.value = 172
    op = operator_card
    op.mu0 = 5.0
    op.mu2grid = mu2grid
    # lhapdf import (maybe i have to dump with a x*), do plots)
    with lhapdf_path(test_pdf):
        pdf = lhapdf.mkPDF("myCT14llo_NF3", 0)
    with cd(tmp_path):
        ev_p.evolve_pdfs(
            [pdf],
            theory,
            op,
            name="EvPDF",
            info_update={"SetDesc": "MyEvolvedPDF", "MZ": 0.2, "Debug": "Debug"},
        )
    with lhapdf_path(tmp_path):
        all_blocks = (load.load_blocks_from_file("EvPDF", 0))[1]
        info = load.load_info_from_file("EvPDF")
        ev_pdf = lhapdf.mkPDF("EvPDF", 0)
    assert info["XMin"] == op.rotations.xgrid.raw[0]
    assert info["SetDesc"] == "MyEvolvedPDF"
    assert info["MZ"] == theory.couplings.alphas.scale
    assert info["Debug"] == "Debug"
    xgrid = op.rotations.xgrid.raw
    for Q2 in [20.0, 100.0, 10000.0]:
        for x in xgrid[10:40]:
            for pid in [21, 1, -1, 2, -2, 3, -3]:
                np.testing.assert_allclose(
                    pdf.xfxQ2(pid, x, Q2),
                    all_blocks[0]["data"][
                        mu2grid.index(Q2) + xgrid.tolist().index(x) * len(mu2grid)
                    ][br.flavor_basis_pids.index(pid)],
                    rtol=1e-2,
                )
                np.testing.assert_allclose(
                    pdf.xfxQ2(pid, x, Q2),
                    ev_pdf.xfxQ2(pid, x, Q2),
                    rtol=1e-2,
                )


@pytest.mark.isolated
def benchmark_evolve_more_members(
    tmp_path, cd, lhapdf_path, theory_card: TheoryCard, operator_card: OperatorCard
):
    theory = theory_card
    theory.order = (1, 0)
    op = operator_card
    operator_card.mu0 = 1.0
    operator_card.mu2grid = [10, 100]
    operator_card.rotations.xgrid = XGrid([1e-7, 0.01, 0.1, 0.2, 0.3])
    with lhapdf_path(test_pdf):
        pdfs = lhapdf.mkPDFs("myMSTW2008nlo90cl")
    d = tmp_path / "sub"
    d.mkdir()
    with lhapdf_path(d):
        with cd(tmp_path):
            ev_p.evolve_pdfs(pdfs, theory, op, install=True, name="Debug")
        # ev_pdfs
        new_pdfs = lhapdf.mkPDFs("Debug")
        new_pdf_1 = lhapdf.mkPDF("Debug", 0)
        new_pdf_2 = lhapdf.mkPDF("Debug", 1)
        info = load.load_info_from_file("Debug")
    assert info["XMin"] == op.rotations.xgrid.raw[0]
    assert len(pdfs) == len(new_pdfs)
    for Q2 in [10, 100]:
        for x in [1e-7, 0.01, 0.1, 0.2, 0.3]:
            for pid in [21, 1, 2]:
                assert new_pdf_1.xfxQ2(pid, x, Q2) != new_pdf_2.xfxQ2(pid, x, Q2)
