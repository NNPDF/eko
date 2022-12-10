import numpy as np
from banana import toy

import eko
import eko.io.legacy as out
from eko.interpolation import XGrid
from eko.io import runcards
from ekobox import cards
from ekobox import evol_pdf as ev_p

op = cards.example.operator()
op.mu0 = 1.65
op.rotations.xgrid = XGrid([0.1, 0.5, 1.0])
op.rotations.pids = np.array([0, 1])
op.configs.interpolation_polynomial_degree = 1
theory = cards.generate_theory(0)


def test_evolve_pdfs_run(fake_lhapdf, cd):
    n = "test_evolve_pdfs_run"
    mytmp = fake_lhapdf / "install"
    mytmp.mkdir()
    store_path = mytmp / "test.tar"
    nt, no = runcards.update(theory, op)
    with cd(mytmp):
        ev_p.evolve_pdfs(
            [toy.mkPDF("", 0)], nt, no, install=True, name=n, store_path=store_path
        )
    p = fake_lhapdf / n
    assert p.exists()
    # check dumped eko
    assert store_path.exists()
    assert store_path.is_file()
    out.load_tar(store_path)


def test_evolve_pdfs_dump_path(fake_lhapdf, cd):
    n = "test_evolve_pdfs_dump_path"
    nt, no = runcards.update(theory, op)
    peko = fake_lhapdf / ev_p.ekofileid(nt, no)
    out.dump_tar(eko.solve(nt, no), peko)
    assert peko.exists()
    with cd(fake_lhapdf):
        ev_p.evolve_pdfs([toy.mkPDF("", 0)], nt, no, name=n, path=fake_lhapdf)
    p = fake_lhapdf / n
    assert p.exists()


def test_evolve_pdfs_dump_file(fake_lhapdf, cd):
    n = "test_evolve_pdfs_dump_file"
    nt, no = runcards.update(theory, op)
    peko = fake_lhapdf / ev_p.ekofileid(nt, no)
    out.dump_tar(eko.solve(nt, no), peko)
    assert peko.exists()
    with cd(fake_lhapdf):
        ev_p.evolve_pdfs([toy.mkPDF("", 0)], nt, no, name=n, path=peko)
    p = fake_lhapdf / n
    assert p.exists()
