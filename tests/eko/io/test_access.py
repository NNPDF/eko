import pytest

from eko.io import access


def test_writeable(tmp_path):
    c = access.AccessConfigs(tmp_path, False, True)
    assert c.read
    c.assert_open()
    assert c.write
    c.assert_writeable()


def test_readable(tmp_path):
    c = access.AccessConfigs(tmp_path, True, True)
    assert c.read
    c.assert_open()
    assert not c.write
    with pytest.raises(access.ReadOnlyOperator):
        c.assert_writeable()


def test_closed(tmp_path):
    for ro in [True, False]:
        c = access.AccessConfigs(tmp_path, ro, False)
        assert not c.read
        with pytest.raises(access.ClosedOperator):
            c.assert_open()
        assert not c.write
        with pytest.raises(access.ClosedOperator):
            c.assert_writeable()
