from banana import toy

import eko
from eko import EKO
from eko.interpolation import XGrid
from ekobox import cards
from ekobox import evol_pdf as ev_p

op = cards.example.operator()
op.mu0 = 1.65
op.rotations.xgrid = XGrid([0.1, 0.5, 1.0])
op.configs.interpolation_polynomial_degree = 1
theory = cards.example.theory()
theory.order = (1, 0)


def test_evolve_pdfs_run(fake_lhapdf, cd):
    n = "test_evolve_pdfs_run"
    mytmp = fake_lhapdf / "install"
    mytmp.mkdir()
    store_path = mytmp / "test.tar"
    with cd(mytmp):
        ev_p.evolve_pdfs(
            [toy.mkPDF("", 0)], theory, op, install=True, name=n, store_path=store_path
        )
    p = fake_lhapdf / n
    assert p.exists()
    # check dumped eko
    assert store_path.exists()
    assert store_path.is_file()
    with EKO.read(store_path):
        pass


def test_evolve_pdfs_dump_path(fake_lhapdf, cd):
    n = "test_evolve_pdfs_dump_path"
    peko = fake_lhapdf / ev_p.DEFAULT_NAME
    eko.solve(theory, op, peko)
    assert peko.exists()
    with cd(fake_lhapdf):
        ev_p.evolve_pdfs([toy.mkPDF("", 0)], theory, op, name=n, path=fake_lhapdf)
    p = fake_lhapdf / n
    assert p.exists()


def test_evolve_pdfs_dump_file(fake_lhapdf, cd):
    n = "test_evolve_pdfs_dump_file"
    peko = fake_lhapdf / ev_p.DEFAULT_NAME
    eko.solve(theory, op, peko)
    assert peko.exists()
    with cd(fake_lhapdf):
        ev_p.evolve_pdfs([toy.mkPDF("", 0)], theory, op, name=n, path=peko)
    p = fake_lhapdf / n
    assert p.exists()
