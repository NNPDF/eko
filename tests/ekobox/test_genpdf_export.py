import numpy as np
import yaml

from ekobox import genpdf
from ekobox.genpdf.parser import LhapdfDataBlock, LhapdfDataFile


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
    g = genpdf.export.dump_info(f, i)
    assert f.exists()
    assert f == g


def fake_blocks(n_blocks, n_x, n_q2, n_pids):
    bs = []
    for _ in range(n_blocks):
        bs.append(
            LhapdfDataBlock(
                xgrid=np.linspace(0.0, 1.0, n_x),
                qgrid=np.geomspace(1.0, 1e3, n_q2),
                pids=np.arange(n_pids),
                data=np.random.rand(n_x * n_q2, n_pids),
            )
        )
    return LhapdfDataFile(header={}, blocks=bs)


def test_dump_set(tmp_path):
    n = "test"
    p = tmp_path / n
    i = {"a": "b", "c": 2}
    nmem = 2
    for pdf_type_list in (None, [{"Bla": "Blub"}] * nmem):
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
