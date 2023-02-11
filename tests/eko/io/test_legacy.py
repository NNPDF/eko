import pytest

from eko.io import legacy


def test_op5to4():
    with pytest.raises(RuntimeError, match="not found"):
        legacy.op5to4([], {})

    mu2 = 1.959
    op = 29348.2342
    err = 54225.24
    op4s = legacy.op5to4([mu2], {legacy.OPERATOR: [op], legacy.ERROR: [err]})
    assert mu2 in op4s
    assert op4s[mu2].operator == op
    assert op4s[mu2].error == err

    op4s_noerr = legacy.op5to4([mu2], {legacy.OPERATOR: [op], legacy.ERROR: None})
    assert op4s_noerr[mu2].error is None
