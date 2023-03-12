from eko.io.types import ReferenceRunning


def test_reference_typed():
    Ref = ReferenceRunning[str]

    val = "ciao"
    scale = 30.0
    r1 = Ref.typed(val, scale)
    r2 = Ref([val, scale])

    assert r1 == r2
