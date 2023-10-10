import numpy as np
from banana import toy

import eko
from eko import EKO
from eko import basis_rotation as br
from eko.interpolation import XGrid
from ekobox import cards
from ekobox import evol_pdf as ev_p


def init_cards():
    op = cards.example.operator()
    op.mu0 = 1.65
    op.xgrid = XGrid([0.1, 0.5, 1.0])
    op.configs.interpolation_polynomial_degree = 1
    theory = cards.example.theory()
    theory.order = (1, 0)
    return theory, op


def test_evolve_pdfs_run(fake_lhapdf, cd):
    theory, op = init_cards()
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
    theory, op = init_cards()
    n = "test_evolve_pdfs_dump_path"
    peko = fake_lhapdf / ev_p.DEFAULT_NAME
    eko.solve(theory, op, peko)
    assert peko.exists()
    with cd(fake_lhapdf):
        ev_p.evolve_pdfs([toy.mkPDF("", 0)], theory, op, name=n, path=fake_lhapdf)
    p = fake_lhapdf / n
    assert p.exists()


def test_evolve_pdfs_dump_file(fake_lhapdf, cd):
    theory, op = init_cards()
    n = "test_evolve_pdfs_dump_file"
    peko = fake_lhapdf / ev_p.DEFAULT_NAME
    eko.solve(theory, op, peko)
    assert peko.exists()
    with cd(fake_lhapdf):
        ev_p.evolve_pdfs([toy.mkPDF("", 0)], theory, op, name=n, path=peko)
    p = fake_lhapdf / n
    assert p.exists()


def test_regroup_evolgrid():
    # basic
    i = [(3.0, 3), (4.0, 3)]
    o = ev_p.regroup_evolgrid(i)
    assert len(o.keys()) == 1
    assert 3 in o
    assert len(o[3]) == 2
    # more advanced
    i = [(4.0, 3), (3.0, 3), (4.0, 4), (3.0, 4)]
    o = ev_p.regroup_evolgrid(i)
    assert len(o.keys()) == 2
    assert 3 in o
    assert 4 in o
    assert len(o[3]) == 2
    assert len(o[4]) == 2
    np.testing.assert_allclose(o[3], o[4])
    # messed up
    i = [(4.0, 3), (4.0, 4), (3.0, 4), (3.0, 3), (5.0, 5)]
    o = ev_p.regroup_evolgrid(i)
    assert len(o.keys()) == 3
    assert 3 in o
    assert 4 in o
    assert 5 in o
    assert len(o[3]) == 2
    assert len(o[4]) == 2
    assert len(o[5]) == 1
    np.testing.assert_allclose(o[3], o[4])


def test_collect_blocks():
    xgrid = [0.1, 0.5, 0.1]

    def mk(eps):
        f = {}
        for ep in eps:
            f[ep] = {
                "pdfs": {
                    pid: np.random.rand(len(xgrid)) for pid in br.flavor_basis_pids
                }
            }
        return f

    # basic
    eps = [(3.0, 3), (4.0, 3)]
    bs = ev_p.collect_blocks(mk(eps), ev_p.regroup_evolgrid(eps), xgrid)
    assert len(bs) == 1
    np.testing.assert_allclose(bs[0]["mu2grid"], (3.0, 4.0))
    # more advanced
    eps = [(4.0, 3), (3.0, 3), (5.0, 4), (3.0, 4)]
    bs = ev_p.collect_blocks(mk(eps), ev_p.regroup_evolgrid(eps), xgrid)
    assert len(bs) == 2
    np.testing.assert_allclose(bs[0]["mu2grid"], (3.0, 4.0))
    np.testing.assert_allclose(bs[1]["mu2grid"], (3.0, 5.0))
