from eko.quantities.couplings import CouplingsInfo


def test_couplings_ref():
    scale = 90.0
    d = dict(alphas=0.1, alphaem=0.01, scale=scale, num_flavs_ref=5)
    couplings = CouplingsInfo.from_dict(d)
    assert couplings.scale == scale
    assert not couplings.em_running
