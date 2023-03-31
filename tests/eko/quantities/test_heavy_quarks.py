from eko.quantities.heavy_quarks import (
    HeavyQuarkMasses,
    HeavyQuarks,
    MatchingRatios,
    QuarkMassRef,
)


def test_heavy_quarks():
    Labels = HeavyQuarks[str]

    c = "charm"
    b = "bottom"
    t = "top"

    labels = Labels([c, b, t])
    assert labels.c == c
    assert labels.b == b
    assert labels.t == t

    new_quark = "fantastique"
    labels.b = new_quark
    assert labels.b == new_quark


def test_concrete_types():
    masses = HeavyQuarkMasses([QuarkMassRef([m, m]) for m in [1.0, 5.0, 100.0]])
    assert all(hqm.value == hqm.scale for hqm in masses)

    ratios = MatchingRatios([1.0, 2.0, 3.0])
    assert ratios.b == 2.0
