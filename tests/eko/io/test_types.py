from math import isnan, nan

from eko.io.types import CouplingsRef, ReferenceRunning


def test_reference_typed():
    Ref = ReferenceRunning[str]

    val = "ciao"
    scale = 30.0
    r1 = Ref.typed(val, scale)
    r2 = Ref([val, scale])

    assert r1 == r2
    assert r1.value == r2.value == val
    assert r1.scale == r2.scale == scale

    new_scale = 134.4323
    r2.scale = new_scale
    assert r2.scale == new_scale == r2[1]


def test_couplings_ref():
    scale = 90.0
    d = dict(
        alphas=[0.1, scale], alphaem=[0.01, nan], max_num_flavs=6, num_flavs_ref=None
    )
    couplings = CouplingsRef.from_dict(d)
    assert couplings.alphas.scale == scale
    assert isnan(couplings.alphaem.scale)
