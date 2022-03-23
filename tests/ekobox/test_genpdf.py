# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ekobox import genpdf


def test_genpdf_exceptions(tmp_path, cd):
    with cd(tmp_path):
        # wrong label
        with pytest.raises(TypeError):
            genpdf.generate_pdf(
                "test_genpdf_exceptions1",
                ["f"],
                {
                    21: lambda x, Q2: 3.0 * x * (1.0 - x),
                    2: lambda x, Q2: 4.0 * x * (1.0 - x),
                },
            )
        # wrong parent pdf
        with pytest.raises(ValueError):
            genpdf.generate_pdf(
                "test_genpdf_exceptions2",
                ["g"],
                10,
            )
        # non-existant PDF set
        with pytest.raises(FileNotFoundError):
            genpdf.install_pdf("foo")
        with pytest.raises(TypeError):
            genpdf.generate_pdf("debug", [21], info_update=(10, 15, 20))


def test_generate_block():
    xg = np.linspace(0.0, 1.0, 5)
    q2s = np.geomspace(1.0, 1e3, 5)
    pids = np.arange(3)
    b = genpdf.generate_block(lambda pid, x, q2: pid * x * q2, xg, q2s, pids)
    assert isinstance(b, dict)
    assert sorted(b.keys()) == sorted(["data", "Q2grid", "xgrid", "pids"])
    assert isinstance(b["data"], np.ndarray)
    assert b["data"].shape == (len(xg) * len(q2s), len(pids))


def test_install_pdf(fake_lhapdf, tmp_path, cd):
    # move into subdir to be able to move
    mytmp = tmp_path / "install"
    mytmp.mkdir()
    n = "test_install_pdf"
    p = mytmp / n
    i = "test.info"
    with cd(mytmp):
        with pytest.raises(FileNotFoundError):
            genpdf.install_pdf(p)
        p.mkdir()
        (p / i).write_text("Bla")
        genpdf.install_pdf(p)
    pp = tmp_path / n
    assert not p.exists()
    assert pp.exists()
    ppi = pp / i
    assert ppi.exists()
    assert "Bla" == ppi.read_text()


def test_generate_pdf_debug(fake_lhapdf, tmp_path, cd):
    mytmp = tmp_path / "install"
    mytmp.mkdir()
    n = "test_generate_pdf_debug"
    xg = np.linspace(0.0, 1.0, 5)
    q2s = np.geomspace(1.0, 1e3, 5)
    p = mytmp / n
    i = f"{n}.info"
    with cd(mytmp):
        genpdf.generate_pdf(
            n,
            [21],
            None,
            info_update={"Debug": "debug"},
            install=True,
            xgrid=xg,
            Q2grid=q2s,
        )
    pp = tmp_path / n
    assert not p.exists()
    assert pp.exists()
    # check info file
    ppi = pp / i
    assert ppi.exists()
    assert "Debug: debug" in ppi.read_text()
    ii = genpdf.load.load_info_from_file(n)
    assert "Debug" in ii
    assert ii["Debug"] == "debug"
    # check member file
    ppm = pp / f"{n}_0000.dat"
    assert ppm.exists()
    assert "PdfType: central" in ppm.read_text()
    head, blocks = genpdf.load.load_blocks_from_file(n, 0)
    assert "PdfType: central" in head
    assert len(blocks) == 1
    b = blocks[0]
    assert 21 in b["pids"]
    for k, line in enumerate(b["data"]):
        for pid, f in zip(b["pids"], line):
            # the gluon is non-zero in the bulk
            if pid == 21 and k > len(xg) - 1 and k < len(b["data"]) - len(xg):
                assert not f == 0.0
            else:
                assert f == 0.0
