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
    mugrid = [(3.0, 4), (10.0, 5), (100.0, 5)]
    theory = theory_card
    theory.order = (1, 0)
    theory.couplings.alphas = 0.118000
    theory.couplings.ref = (91.1876, 5)
    theory.couplings.alphaem = 0.007496
    theory.heavy.masses.c.value = 1.3
    theory.heavy.masses.b.value = 4.75
    theory.heavy.masses.t.value = 172
    op = operator_card
    op.init = (5.0, 4)
    op.mugrid = mugrid
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
            store_path=tmp_path / "eko.tar",
        )
    with lhapdf_path(tmp_path):
        all_blocks = (load.load_blocks_from_file("EvPDF", 0))[1]
        info = load.load_info_from_file("EvPDF")
        ev_pdf = lhapdf.mkPDF("EvPDF", 0)
    assert info["XMin"] == op.xgrid.raw[0]
    assert info["SetDesc"] == "MyEvolvedPDF"
    assert info["MZ"] == theory.couplings.ref[0]
    assert info["Debug"] == "Debug"
    xgrid = op.xgrid.raw
    for idx, mu2 in enumerate(op.mu2grid):
        for x in xgrid[10:40]:
            for pid in [21, 1, -1, 2, -2, 3, -3]:
                np.testing.assert_allclose(
                    pdf.xfxQ2(pid, x, mu2),
                    all_blocks[0]["data"][idx + xgrid.tolist().index(x) * len(mugrid)][
                        br.flavor_basis_pids.index(pid)
                    ],
                    rtol=1e-2,
                )
                np.testing.assert_allclose(
                    pdf.xfxQ2(pid, x, mu2),
                    ev_pdf.xfxQ2(pid, x, mu2),
                    rtol=1e-2,
                )


@pytest.mark.isolated
def benchmark_evolve_more_members(
    tmp_path, cd, lhapdf_path, theory_card: TheoryCard, operator_card: OperatorCard
):
    theory = theory_card
    theory.order = (1, 0)
    op = operator_card
    op.init = (1.0, 3)
    op.mugrid = [(10.0, 5), (100.0, 5)]
    op.xgrid = XGrid([1e-7, 0.01, 0.1, 0.2, 0.3])
    with lhapdf_path(test_pdf):
        pdfs = lhapdf.mkPDFs("myMSTW2008nlo90cl")
    d = tmp_path / "sub"
    d.mkdir()
    with lhapdf_path(d):
        with cd(tmp_path):
            ev_p.evolve_pdfs(
                pdfs,
                theory,
                op,
                install=True,
                name="Debug",
                store_path=tmp_path / "eko.tar",
            )
        # ev_pdfs
        new_pdfs = lhapdf.mkPDFs("Debug")
        new_pdf_1 = lhapdf.mkPDF("Debug", 0)
        new_pdf_2 = lhapdf.mkPDF("Debug", 1)
        info = load.load_info_from_file("Debug")
    assert info["XMin"] == op.xgrid.raw[0]
    assert len(pdfs) == len(new_pdfs)
    for mu2 in [10, 100]:
        for x in [1e-7, 0.01, 0.1, 0.2, 0.3]:
            for pid in [21, 1, 2]:
                assert new_pdf_1.xfxQ2(pid, x, mu2) != new_pdf_2.xfxQ2(pid, x, mu2)
