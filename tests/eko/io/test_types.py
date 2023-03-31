from eko.io.types import ReferenceRunning


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
