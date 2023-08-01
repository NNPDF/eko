from io import StringIO

import numpy as np
import pytest
import yaml

from eko.io import runcards as rc
from eko.io.bases import Bases


def test_flavored_mu2grid():
    mugrid = list(range(5, 40, 5))
    masses = [10, 20, 30]
    ratios = [1, 1, 1]

    flavored = rc.flavored_mugrid(mugrid, masses, ratios)
    assert pytest.approx([flav for _, flav in flavored]) == [3, 4, 4, 5, 5, 6, 6]

    # check we can dump
    stream = StringIO()
    _ = yaml.safe_dump(flavored, stream)
    stream.seek(0)
    deser = yaml.safe_load(stream)
    assert len(flavored) == len(deser)
    # after deserialization one is now list instead of tuple
    for t, l in zip(flavored, deser):
        assert len(t) == len(l)
        assert t == tuple(l)


def check_dumpable(no):
    """Check we can write and read to yaml."""
    so = StringIO()
    yaml.dump(no.raw, so)
    so.seek(0)
    noo = yaml.safe_load(so)


def test_legacy_fallback():
    assert rc.Legacy.fallback(1, 2, 3) == 1
    assert rc.Legacy.fallback(None, 2, 3) == 2
    assert rc.Legacy.fallback(None, 2, 3, default=7) == 2
    assert rc.Legacy.fallback(None, None, 3) == 3
    assert rc.Legacy.fallback(None, None, None) is None
    assert rc.Legacy.fallback(None, None, None, default=7) == 7
