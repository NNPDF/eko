import lhapdf
import numpy as np
import pytest
from banana.data.genpdf import load
from utils import cd, lhapdf_path, test_pdf

from ekobox import evol_pdf as ev_p
from ekobox import gen_info as g_i
from ekobox import gen_op as g_o
from ekobox import gen_theory as g_t


def test_evolve_single_member(tmp_path):
    op = g_o.gen_op_card(
        [10, 100], update={"interpolation_xgrid": [1e-7, 0.01, 0.1, 0.2, 0.3]}
    )
    theory = g_t.gen_theory_card(0, 1.0)
    with lhapdf_path(test_pdf):
        pdf = lhapdf.mkPDF("myCT14llo_NF3", 0)
    with cd(tmp_path):
        ev_p.evolve_PDFs(
            [pdf],
            theory,
            op,
            info_update={"SetDesc": "MyEvolvedPDF", "MZ": 0.2, "Debug": "Debug"},
        )
    with lhapdf_path(tmp_path):
        all_blocks = (load.load_blocks_from_file("Evolved_PDF", 0))[1]
        info = load.load_info_from_file("Evolved_PDF")
    assert info["XMin"] == op["interpolation_xgrid"][0]
    assert info["SetDesc"] == "MyEvolvedPDF"
    assert info["MZ"] == theory["MZ"]
    assert info["Debug"] == "Debug"


def test_evolve_more_members(tmp_path):
    op = g_o.gen_op_card(
        [10, 100], update={"interpolation_xgrid": [1e-7, 0.01, 0.1, 0.2, 0.3]}
    )
    theory = g_t.gen_theory_card(0, 1.0)
    with lhapdf_path(test_pdf):
        pdfs = lhapdf.mkPDFs("myMSTW2008nlo90cl")
    d = tmp_path / "sub"
    d.mkdir()
    with lhapdf_path(d):
        with cd(tmp_path):
            ev_p.evolve_PDFs(pdfs, theory, op, install=True, name="Debug")
        all_blocks = (load.load_blocks_from_file("Debug", 1))[1]
        info = load.load_info_from_file("Debug")
    assert info["XMin"] == op["interpolation_xgrid"][0]
