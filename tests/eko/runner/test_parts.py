from eko.runner import parts


def test_evolve_configs(eko_factory):
    # QCD@LO
    e10 = eko_factory.get()
    assert e10.theory_card.order == (1, 0)
    p10 = parts.evolve_configs(e10)
    assert p10["matching_order"] == (0, 0)
    # QCD@N3LO + QED@N2LO w/o matching_order
    eko_factory.theory.order = (4, 3)
    eko_factory.theory.matching_order = None
    e43 = eko_factory.get({})
    assert e43.theory_card.order == (4, 3)
    p43 = parts.evolve_configs(e43)
    assert p43["matching_order"] == (3, 0)
    # QCD@N3LO + QED@N2LO w/ matching_order
    eko_factory.theory.matching_order = (3, 0)
    e43b = eko_factory.get({})
    assert e43b.theory_card.order == (4, 3)
    p43b = parts.evolve_configs(e43b)
    assert p43b["matching_order"] == (3, 0)
