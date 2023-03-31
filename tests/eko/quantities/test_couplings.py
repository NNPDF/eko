from eko.quantities.couplings import CouplingsInfo


def test_couplings_ref():
    scale = 90.0
    d = dict(alphas=0.1, alphaem=0.01, scale=scale, max_num_flavs=6, num_flavs_ref=None)
    couplings = CouplingsInfo.from_dict(d)
    assert couplings.scale == scale
    assert not couplings.em_running
