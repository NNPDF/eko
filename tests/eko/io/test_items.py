import io

import lz4.frame
import numpy as np
import pytest

from eko.io import struct


class TestOperator:
    def test_value_only(self):
        v = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        assert opv.error is None
        stream = io.BytesIO()
        opv.save(stream)
        stream.seek(0)
        opv_ = struct.Operator.load(stream)
        np.testing.assert_allclose(opv.operator, opv_.operator)
        np.testing.assert_allclose(v, opv_.operator)
        assert opv_.error is None

    def test_value_and_error(self):
        v, e = np.random.rand(2, 2, 2)
        opve = struct.Operator(operator=v, error=e)
        stream = io.BytesIO()
        opve.save(stream)
        stream.seek(0)
        opve_ = struct.Operator.load(stream)
        np.testing.assert_allclose(opve.operator, opve_.operator)
        np.testing.assert_allclose(v, opve_.operator)
        np.testing.assert_allclose(opve.error, opve_.error)
        np.testing.assert_allclose(e, opve_.error)

    def test_load_error_is_not_lz4(self, monkeypatch):
        stream = io.BytesIO()
        with pytest.raises(RuntimeError, match="LZ4"):
            struct.Operator.load(stream)
