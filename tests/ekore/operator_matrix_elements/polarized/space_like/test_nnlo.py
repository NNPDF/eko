# Test NNLO polarized OME
import numpy as np

from ekore.harmonics import cache as c
from ekore.operator_matrix_elements.polarized.space_like import as2

NF = 4


def test_A_1_shape():
    N = 2
    L = 3.0
    cache = c.reset()
    aS2i = as2.A_singlet(N, cache, L, NF)
    aS2nsi = as2.A_ns(N, cache, L)
    assert aS2i.shape == (3, 3)
    assert aS2nsi.shape == (2, 2)


def test_quark_number_conservation():
    L = 10
    N = 1.0
    cache = c.reset()
    np.testing.assert_allclose(as2.A_qq_ns(N, cache, L), 0.0, atol=1e-13)


def test_hg():
    refs = {
        0: [
            -242.3869306886845,
            -480.6122566782359,
            -814.6150038529145,
            -1244.9445223204148,
            -1771.5995558739958,
        ],
        10: [
            -21.21409118251165,
            -295.28287396218656,
            -664.653836052461,
            -1121.4595082763883,
            -1667.8336593381266,
        ],
    }
    for L, vals in refs.items():
        test_vals = []
        for i, _ in enumerate(vals):
            n = 2 * i + 3
            cache = c.reset()
            test_vals.append(as2.A_hg(n, cache, L))
        np.testing.assert_allclose(test_vals, vals)


def test_hq():
    refs = {
        0: [
            -0.4642489711934156,
            -0.23400164609053498,
            -0.13602229539775093,
            -0.08807148818984994,
            -0.061448213442336426,
        ],
        10: [
            -24.229681069958847,
            -10.410791769547325,
            -5.706357572365681,
            -3.587147848500778,
            -2.459209109825822,
        ],
    }
    zqq_shift = [
        -0.30864197530864196,
        -0.10508641975308641,
        -0.044825072886297376,
        -0.022855052583447645,
        -0.013143371270093034,
    ]
    for L, vals in refs.items():
        test_vals = []
        for i, _ in enumerate(vals):
            n = 2 * i + 3
            cache = c.reset()
            test_vals.append(as2.A_hq_ps(n, cache, L, NF))
        np.testing.assert_allclose(test_vals, np.array(vals) + zqq_shift)


def test_gq():
    L = 10
    refs = {
        0: [
            1.867283950617284,
            1.1509958847736625,
            0.8681239876903142,
            0.7156917748352198,
            0.619192169290155,
        ],
        10: [
            82.1141975308642,
            49.13124279835391,
            35.44635528020732,
            27.92964435175076,
            23.15793214237356,
        ],
    }
    vals = []
    for L, vals in refs.items():
        test_vals = []
        for i, _ in enumerate(vals):
            n = 2 * i + 3
            cache = c.reset()
            test_vals.append(as2.A_gq(n, cache, L))
        np.testing.assert_allclose(test_vals, np.array(vals))


def test_gg():
    L = 10
    refs = {
        0: [
            -27.099794238683128,
            -34.67929053497942,
            -39.23791305850572,
            -42.52361109441795,
            -45.102624484723364,
        ],
        10: [
            -522.9022633744856,
            -683.3449695473251,
            -780.8484977685803,
            -851.6504358380329,
            -907.5255285880683,
        ],
    }
    vals = []
    for L, vals in refs.items():
        test_vals = []
        for i, _ in enumerate(vals):
            n = 2 * i + 3
            cache = c.reset()
            test_vals.append(as2.A_gg(n, cache, L))
        np.testing.assert_allclose(test_vals, np.array(vals))


def test_qq():
    L = 10
    refs = {
        0: [
            -5.057098765432098,
            -7.5125843621399175,
            -9.175463846296896,
            -10.441618105018478,
            -11.466735432197467,
        ],
        10: [
            -139.0077160493827,
            -200.2730781893004,
            -240.83263189870303,
            -271.2800772746946,
            -295.68289529001197,
        ],
    }
    vals = []
    for L, vals in refs.items():
        test_vals = []
        for i, _ in enumerate(vals):
            n = 2 * i + 3
            cache = c.reset()
            test_vals.append(as2.A_qq_ns(n, cache, L))
        np.testing.assert_allclose(test_vals, np.array(vals))
