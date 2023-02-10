import pathlib

import pytest

from eko.io import exceptions


def test_oplocerr():
    p = pathlib.Path.home() / "path" / "to" / "missing" / "eko.tar"
    with pytest.raises(exceptions.OperatorLocationError, match="not-found"):
        raise exceptions.OperatorLocationError(p)
