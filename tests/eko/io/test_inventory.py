from dataclasses import dataclass

import numpy as np
import pytest

from eko.io import access, inventory, items


@dataclass(frozen=True)
class FakeH(items.Header):
    blub: int


def test_contentless(tmp_path):
    acw = inventory.AccessConfigs(tmp_path, False, True)
    iw = inventory.Inventory(tmp_path, acw, FakeH, contentless=True, name="Bla")
    assert "Bla" in str(iw)
    one = FakeH(1)
    assert one not in iw
    assert len(list(tmp_path.glob("*"))) == 0
    # set an element
    iw[one] = "blub"
    assert one in iw
    assert iw[one] is None
    ls = list(tmp_path.glob("*"))
    assert len(iw) == 1
    assert len(ls) == 1
    assert ls[0].suffix == inventory.HEADER_EXT
    # let's make a second read-only instance
    acr = inventory.AccessConfigs(tmp_path, True, True)
    ir = inventory.Inventory(tmp_path, acr, FakeH, contentless=True, name="noihaha")
    assert "Bla" not in str(ir)
    assert one not in ir
    assert ir[one] is None
    # deletion is empty for contentless
    del iw[one]
    assert one in iw
    assert iw[one] is None
    assert len(list(tmp_path.glob("*"))) == 1
    iw.empty()
    assert len(list(tmp_path.glob("*"))) == 1
    # non-existant
    two = FakeH(2)
    assert two not in iw
    with pytest.raises(inventory.LookupError):
        iw[two]


def test_contentfull(tmp_path):
    def o():
        return items.Operator(np.random.rand(2, 2, 2, 2))

    acw = inventory.AccessConfigs(tmp_path, False, True)
    iw = inventory.Inventory(tmp_path, acw, FakeH, name="Bla")
    one = FakeH(1)
    assert one not in iw
    assert len(list(tmp_path.glob("*"))) == 0
    # set an element
    o_one = o()
    iw[one] = o_one
    assert one in iw
    assert iw[one] is not None
    assert len(iw) == 1
    assert len(list(tmp_path.glob("*" + inventory.HEADER_EXT))) == 1
    assert len(list(tmp_path.glob("*" + inventory.COMPRESSED_EXT))) == 1
    # let's make a second read-only instance
    acr = inventory.AccessConfigs(tmp_path, True, True)
    ir = inventory.Inventory(tmp_path, acr, FakeH, name="noihaha")
    assert "Bla" not in str(ir)
    assert one not in ir
    # if we sync we know the header is there
    ~ir
    assert one in ir
    assert len(ir) == 1
    # do an actual read to load
    np.testing.assert_allclose(ir[one].operator, o_one.operator)
    # but we can not write
    with pytest.raises(access.ReadOnlyOperator):
        ir[one] = o()
    # let's overwrite the op in the writable, so they diverge
    o_one_p = o()
    iw[one] = o_one_p
    assert len(iw) == 1
    np.testing.assert_allclose(iw.cache[one].operator, o_one_p.operator)
    np.testing.assert_allclose(ir.cache[one].operator, o_one.operator)
    # let's actually read again
    ~ir
    np.testing.assert_allclose(ir[one].operator, o_one_p.operator)
    # non-existant
    two = FakeH(2)
    assert two not in ir
    with pytest.raises(inventory.LookupError):
        ir[two]
