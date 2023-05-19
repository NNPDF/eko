import numpy as np
import pytest

from eko import basis_rotation as br
from eko.io.runcards import flavored_mugrid
from ekobox import genpdf

MASSES = [1.5, 4.0, 170.0]


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
    mu2s = np.geomspace(1.0, 1e3, 5)
    evolgrid = flavored_mugrid(mu2s.tolist(), MASSES, [1.0, 1.0, 1.0])
    pids = np.arange(3)
    b = genpdf.generate_block(lambda pid, x, q2: pid * x * q2, xg, evolgrid, pids)
    assert isinstance(b, dict)
    assert sorted(b.keys()) == sorted(["data", "mu2grid", "xgrid", "pids"])
    assert isinstance(b["data"], np.ndarray)
    assert b["data"].shape == (len(xg) * len(mu2s), len(pids))


def test_install_pdf(fake_lhapdf, cd):
    # move into subdir to be able to move
    mytmp = fake_lhapdf / "install"
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
    pp = fake_lhapdf / n
    assert not p.exists()
    assert pp.exists()
    ppi = pp / i
    assert ppi.exists()
    assert "Bla" == ppi.read_text()


def test_generate_pdf_debug_pid(fake_lhapdf, cd):
    mytmp = fake_lhapdf / "install"
    mytmp.mkdir()
    n = "test_generate_pdf_debug_pid"
    xg = np.linspace(0.0, 1.0, 5)
    mu2s = np.geomspace(1.0, 1e3, 7)
    evolgrid = flavored_mugrid(mu2s.tolist(), MASSES, [1.0, 1.0, 1.0])
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
            evolgrid=evolgrid,
        )
    pp = fake_lhapdf / n
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
            # the gluon is non-zero in the bulk - x=0 is included here
            if (
                pid == 21
                and k > len(b["mu2grid"]) - 1
                and k < len(b["data"]) - len(b["mu2grid"])
            ):
                assert not f == 0.0
            else:
                assert f == 0.0


def test_generate_pdf_pdf_evol(fake_lhapdf, fake_nn31, fake_mstw, fake_ct14, cd):
    # iterate pdfs with their error type and number of blocks
    for fake_pdf, err_type, nmem, nb in (
        (fake_nn31, "replica", 1, 2),
        (fake_mstw, "error", 1, 3),
        (fake_ct14, "", 0, 1),
    ):
        n = f"test_generate_pdf_{err_type}_evol"
        p = fake_lhapdf / n
        i = f"{n}.info"
        pi = p / i
        with cd(fake_lhapdf):
            genpdf.generate_pdf(
                n,
                ["S"],
                fake_pdf,
                members=nmem > 0,
            )
        assert p.exists()
        # check info file
        assert pi.exists()
        # check member files
        for m in range(nmem + 1):
            pm = p / f"{n}_{m:04d}.dat"
            assert pm.exists()
            head, blocks = genpdf.load.load_blocks_from_file(n, m)
            assert ("PdfType: central" if m == 0 else f"PdfType: {err_type}") in head
            assert len(blocks) == nb
            for b in blocks:
                for k, line in enumerate(b["data"]):
                    for pid, f in zip(b["pids"], line):
                        # the singlet is non-zero in the bulk - x >= 0
                        if abs(pid) in range(1, 6 + 1) and k < len(b["data"]) - len(
                            b["mu2grid"]
                        ):
                            assert not f == 0.0
                            assert not np.isclose(f, 0.0, atol=1e-15)
                        else:
                            # MSTW is not 0 at the end, but only almost
                            if err_type == "error" and k >= len(b["data"]) - len(
                                b["mu2grid"]
                            ):
                                np.testing.assert_allclose(f, 0.0, atol=1e-15)
                            else:
                                assert f == 0.0


def test_generate_pdf_toy_antiqed(fake_lhapdf, cd):
    # iterate pdfs with their error type and number of blocks
    n = "test_generate_pdf_toy_antiqed"
    xg = np.linspace(1e-5, 1.0, 5)
    mu2s = np.geomspace(1.0, 1e3, 7)
    evolgrid = flavored_mugrid(mu2s.tolist(), MASSES, [1.0, 1.0, 1.0])
    anti_qed_singlet = np.zeros_like(br.flavor_basis_pids, dtype=np.float_)
    anti_qed_singlet[br.flavor_basis_pids.index(1)] = -4
    anti_qed_singlet[br.flavor_basis_pids.index(-1)] = -4
    anti_qed_singlet[br.flavor_basis_pids.index(2)] = 1
    anti_qed_singlet[br.flavor_basis_pids.index(-2)] = 1
    p = fake_lhapdf / n
    i = f"{n}.info"
    pi = p / i
    with cd(fake_lhapdf):
        genpdf.generate_pdf(
            n,
            [anti_qed_singlet],
            "toy",
            xgrid=xg,
            evolgrid=evolgrid,
        )
    assert p.exists()
    # check info file
    assert pi.exists()
    # check member files
    pm = p / f"{n}_0000.dat"
    assert pm.exists()
    assert "PdfType: central" in pm.read_text()
    head, blocks = genpdf.load.load_blocks_from_file(n, 0)
    assert "PdfType: central" in head
    assert len(blocks) == 1
    b = blocks[0]
    for k, line in enumerate(b["data"]):
        for pid, f in zip(b["pids"], line):
            # the u and d are non-zero in the bulk - x=0 is not included here
            if abs(pid) in [1, 2] and k < len(b["data"]) - len(b["mu2grid"]):
                assert not f == 0.0
            else:
                assert f == 0.0
