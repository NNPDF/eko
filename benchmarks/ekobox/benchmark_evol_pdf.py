# -*- coding: utf-8 -*-
import pathlib

import numpy as np
import pytest

from eko import basis_rotation as br
from ekobox import evol_pdf as ev_p
from ekobox import operators_card as oc
from ekobox import theory_card as tc
from ekobox.genpdf import load

test_pdf = pathlib.Path(__file__).parent / "fakepdf"
lhapdf = pytest.importorskip("lhapdf")


@pytest.mark.isolated
def benchmark_evolve_single_member(tmp_path, cd, lhapdf_path):
    q2grid = [
        20.0,
        100.0,
        10000.0,
    ]
    op = oc.generate(q2grid)
    theory = tc.generate(
        0,
        5.0,
        update={
            "alphas": 0.118000,
            "alphaqed": 0.007496,
            "QED": 0,
            "MaxNfPdf": 3,
            "MaxNfAs": 3,
            "Qref": 91.1876,
            "mc": 1.3,
            "mb": 4.75,
            "mt": 172,
            "kcThr": 1,
            "kbThr": 1,
            "ktThr": 1,
        },
    )
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
    assert info["XMin"] == op["xgrid"][0]
    assert info["SetDesc"] == "MyEvolvedPDF"
    assert info["MZ"] == theory["MZ"]
    assert info["Debug"] == "Debug"
    xgrid = op["xgrid"]
    for Q2 in [20.0, 100.0, 10000.0]:
        for x in xgrid[10:40]:
            for pid in [21, 1, -1, 2, -2, 3, -3]:
                np.testing.assert_allclose(
                    pdf.xfxQ2(pid, x, Q2),
                    all_blocks[0]["data"][
                        q2grid.index(Q2) + xgrid.index(x) * len(q2grid)
                    ][br.flavor_basis_pids.index(pid)],
                    rtol=1e-2,
                )
                np.testing.assert_allclose(
                    pdf.xfxQ2(pid, x, Q2),
                    ev_pdf.xfxQ2(pid, x, Q2),
                    rtol=1e-2,
                )


@pytest.mark.isolated
def benchmark_evolve_more_members(tmp_path, cd, lhapdf_path):
    op = oc.generate([10, 100], update={"xgrid": [1e-7, 0.01, 0.1, 0.2, 0.3]})
    theory = tc.generate(0, 1.0)
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
    assert info["XMin"] == op["xgrid"][0]
    assert len(pdfs) == len(new_pdfs)
    for Q2 in [10, 100]:
        for x in [1e-7, 0.01, 0.1, 0.2, 0.3]:
            for pid in [21, 1, 2]:
                assert new_pdf_1.xfxQ2(pid, x, Q2) != new_pdf_2.xfxQ2(pid, x, Q2)
