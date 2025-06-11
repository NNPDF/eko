import numpy as np
import yaml

from ekobox import genpdf


def test_list_to_str():
    a = genpdf.export.list_to_str([1, 2])
    assert isinstance(a, str)
    assert "1." in a
    b = genpdf.export.list_to_str([1.0, 2.0])
    assert isinstance(b, str)
    assert "1." in a


def test_array_to_str():
    s = (2, 2)
    a = genpdf.export.array_to_str(np.arange(4).reshape(s))
    assert isinstance(a, str)
    assert "1." in a
    b = np.array([e.split() for e in a.splitlines()])
    assert b.shape == s


def test_dump_info(tmp_path):
    n = "test"
    p = tmp_path / n
    f = p / f"{n}.info"
    i = {"a": "b", "c": 2}
    genpdf.export.dump_info(p, i)
    assert p.exists()
    assert f.exists()
    # the files might not be perfect yaml, but should be yaml compatible
    with open(f, encoding="utf-8") as o:
        ii = yaml.safe_load(o)
    for k, v in i.items():
        assert k in ii
        assert ii[k] == v


def test_dump_info_to_file(tmp_path):
    f = tmp_path / "blub.info"
    i = {"a": "b", "c": 2}
    genpdf.export.dump_info(f, i)
    assert f.exists()


def fake_blocks(n_blocks, n_x, n_q2, n_pids):
    bs = []
    for _ in range(n_blocks):
        bs.append(
            {
                "xgrid": np.linspace(0.0, 1.0, n_x),
                "mu2grid": np.geomspace(1.0, 1e3, n_q2),
                "pids": np.arange(n_pids),
                "data": np.random.rand(n_x * n_q2, n_pids),
            }
        )
    return bs


def test_dump_blocks(tmp_path):
    n = "test"
    p = tmp_path / n
    nb = 2
    for m in range(3):
        f = p / f"{n}_{m:04d}.dat"
        is_my_type = m > 1
        pdf_type = "Bla: blub" if is_my_type else None
        genpdf.export.dump_blocks(p, m, fake_blocks(nb, 2, 2, 2), pdf_type=pdf_type)
        assert p.exists()
        assert f.exists()
        cnt = f.read_text()
        if is_my_type:
            assert "Bla: blub" in cnt
        else:
            assert ("central" in cnt) == (m == 0)
        assert "Format" in cnt
        assert cnt.count("---") == nb + 1


def test_dump_blocks_to_file(tmp_path):
    f = tmp_path / "mem.dat"
    genpdf.export.dump_blocks(f, 0, fake_blocks(2, 2, 2, 2))
    assert f.exists()


def test_dump_set(tmp_path):
    n = "test"
    p = tmp_path / n
    i = {"a": "b", "c": 2}
    nmem = 2
    for pdf_type_list in (None, ["Bla: Blub"] * nmem):
        genpdf.export.dump_set(
            p, i, [fake_blocks(2, 2, 2, 2) for _ in range(nmem)], pdf_type_list
        )
        assert p.exists()
        f = p / f"{n}.info"
        assert f.exists()
        for m in range(nmem):
            f = p / f"{n}_{m:04d}.dat"
            assert f.exists()
            if pdf_type_list is not None:
                assert "Bla: Blub" in f.read_text()
