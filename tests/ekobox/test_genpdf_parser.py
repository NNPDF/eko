import numpy as np

from ekobox.genpdf.parser import LhapdfDataBlock, LhapdfDataFile


def test_genpdf_parser_ct14(fake_lhapdf, fake_ct14, tmp_path):
    # read
    f = LhapdfDataFile.read_with_set(fake_lhapdf, fake_ct14)
    assert "PdfType" in f.header
    assert f.header["PdfType"] == "central"
    assert len(f.blocks) == 1
    b0 = f.blocks[0]
    assert b0.is_valid()
    assert sorted(b0.pids) == sorted([-3, -2, -1, 21, 1, 2, 3])
    assert len(b0.data.T) == 7
    np.testing.assert_allclose(b0.xgrid[0], 1e-9)
    # write
    t = tmp_path / "blub.dat"
    f.write(t)
    # read back
    g = LhapdfDataFile.read(t)
    assert g.header == f.header
    assert len(g.blocks) == len(f.blocks)
    c0 = g.blocks[0]
    np.testing.assert_allclose(c0.xgrid, b0.xgrid)
    np.testing.assert_allclose(c0.qgrid, b0.qgrid)
    np.testing.assert_allclose(c0.pids, b0.pids)
    np.testing.assert_allclose(c0.data, b0.data)


def test_genpdf_parser_mstw(fake_lhapdf, fake_mstw):
    # read
    f0 = LhapdfDataFile.read_with_set(fake_lhapdf, fake_mstw, 0)
    assert "PdfType" in f0.header
    assert f0.header["PdfType"] == "central"
    assert len(f0.blocks) == 3
    for b in f0.blocks:
        assert b.is_valid()
    b0 = f0.blocks[0]
    assert sorted(b0.pids) == sorted([-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5])
    assert len(b0.data.T) == 11
    np.testing.assert_allclose(b0.xgrid[0], 1e-6)
    # check also first member
    f1 = LhapdfDataFile.read_with_set(fake_lhapdf, fake_mstw, 1)
    assert "PdfType" in f1.header
    assert f1.header["PdfType"] == "error"


def test_genpdf_parser_nn31(fake_lhapdf, fake_nn31):
    # read
    f0 = LhapdfDataFile.read_with_set(fake_lhapdf, fake_nn31, 0)
    assert "PdfType" in f0.header
    assert f0.header["PdfType"] == "central"
    assert len(f0.blocks) == 2
    for b in f0.blocks:
        assert b.is_valid()
    b0 = f0.blocks[0]
    assert sorted(b0.pids) == sorted([-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5])
    assert len(b0.data.T) == 11
    np.testing.assert_allclose(b0.xgrid[0], 1e-9)
    # check also first member
    f1 = LhapdfDataFile.read_with_set(fake_lhapdf, fake_nn31, 1)
    assert "PdfType" in f1.header
    assert f1.header["PdfType"] == "replica"


def test_genpdf_parser_block_add():
    a = LhapdfDataBlock(
        xgrid=np.array([0.5]),
        qgrid=np.array([10.0]),
        pids=np.array([1]),
        data=np.array([[1.0], [-1.0]]),
    )
    b = LhapdfDataBlock(
        xgrid=np.array([0.5]),
        qgrid=np.array([10.0]),
        pids=np.array([1]),
        data=np.array([[2.0], [1.0]]),
    )
    c = a.add(b)
    np.testing.assert_allclose(c.xgrid, a.xgrid)
    np.testing.assert_allclose(c.qgrid, a.qgrid)
    np.testing.assert_allclose(c.pids, a.pids)
    np.testing.assert_allclose(c.data, np.array([[3.0], [0.0]]))
    b = LhapdfDataBlock(
        xgrid=np.array([0.5]),
        qgrid=np.array([10.0]),
        pids=np.array([2]),
        data=np.array([[2.0], [1.0]]),
    )
    c = a.add(b)
    np.testing.assert_allclose(c.xgrid, a.xgrid)
    np.testing.assert_allclose(c.qgrid, a.qgrid)
    np.testing.assert_allclose(c.pids, np.array([1, 2]))
    np.testing.assert_allclose(c.data, np.array([[1.0, 2.0], [-1.0, 1.0]]))
    b = LhapdfDataBlock(
        xgrid=np.array([0.5]),
        qgrid=np.array([10.0]),
        pids=np.array([2]),
        data=np.array([[2.0], [1.0]]),
    )
    c = a.add(b)
    np.testing.assert_allclose(c.xgrid, a.xgrid)
    np.testing.assert_allclose(c.qgrid, a.qgrid)
    np.testing.assert_allclose(c.pids, np.array([1, 2]))
    np.testing.assert_allclose(c.data, np.array([[1.0, 2.0], [-1.0, 1.0]]))
    b = LhapdfDataBlock(
        xgrid=np.array([0.5]),
        qgrid=np.array([10.0]),
        pids=np.array([1, 2]),
        data=np.array([[3.0, 2.0], [4.0, 1.0]]),
    )
    c = a.add(b)
    np.testing.assert_allclose(c.xgrid, a.xgrid)
    np.testing.assert_allclose(c.qgrid, a.qgrid)
    np.testing.assert_allclose(c.pids, np.array([1, 2]))
    np.testing.assert_allclose(c.data, np.array([[4.0, 2.0], [3.0, 1.0]]))
    a = LhapdfDataBlock(
        xgrid=np.array([0.5]),
        qgrid=np.array([10.0]),
        pids=np.array([1, -1]),
        data=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    b = LhapdfDataBlock(
        xgrid=np.array([0.5]),
        qgrid=np.array([10.0]),
        pids=np.array([-1, 1]),
        data=np.array([[20.0, 10.0], [40.0, 30.0]]),
    )
    c = a.add(b)
    np.testing.assert_allclose(c.xgrid, a.xgrid)
    np.testing.assert_allclose(c.qgrid, a.qgrid)
    np.testing.assert_allclose(c.pids, np.array([-1, 1]))
    np.testing.assert_allclose(c.data, np.array([[22.0, 11.0], [44.0, 33.0]]))
