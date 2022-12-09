"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.
"""
import numpy as np
import pytest

from eko import compatibility, thresholds
from eko.couplings import Couplings


class TestCouplings:
    def test_from_dict(self):
        theory_dict = {
            "alphas": 0.118,
            "alphaem": 0.00781,
            "Qref": 91.0,
            "nfref": None,
            "Q0": 1,
            "order": (1, 0),
            "ModEv": "EXA",
            "fact_to_ren_scale_ratio": 1.0,
            "mc": 2.0,
            "mb": 4.0,
            "mt": 175.0,
            "kcThr": 1.0,
            "kbThr": 1.0,
            "ktThr": 1.0,
            "MaxNfAs": 6,
            "HQ": "POLE",
            "ModSV": None,
            "alphaem_running": False,
        }
        sc = Couplings.from_dict(theory_dict)
        assert sc.a(theory_dict["Qref"] ** 2)[0] == theory_dict["alphas"] / (
            4.0 * np.pi
        )
        assert sc.a(theory_dict["Qref"] ** 2)[1] == theory_dict["alphaem"] / (
            4.0 * np.pi
        )

    def test_init(self):
        # prepare
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        couplings_ref = np.array([alphas_ref, alphaem_ref])
        scale_ref = 91.0**2
        nf = 4
        threshold_holder = thresholds.ThresholdsAtlas.ffns(nf)
        # create
        sc = Couplings(
            couplings_ref, scale_ref, threshold_holder.area_walls[1:-1], (1.0, 1.0, 1.0)
        )
        assert sc.q2_ref == scale_ref
        assert sc.a_ref[0] == couplings_ref[0] / 4.0 / np.pi
        assert sc.a_ref[1] == couplings_ref[1] / 4.0 / np.pi
        # from theory dict
        for ModEv in ["EXP", "EXA"]:
            for PTOs in range(1, 3 + 1):
                for PTOem in range(2 + 1):
                    setup = dict(
                        alphas=alphas_ref,
                        alphaem=alphaem_ref,
                        Qref=np.sqrt(scale_ref),
                        nfref=None,
                        order=(PTOs, PTOem),
                        ModEv=ModEv,
                        FNS="FFNS",
                        NfFF=nf,
                        Q0=2,
                        fact_to_ren_scale_ratio=1,
                        mc=2.0,
                        mb=4.0,
                        mt=175.0,
                        kcThr=1.0,
                        kbThr=1.0,
                        ktThr=1.0,
                        MaxNfAs=6,
                        HQ="POLE",
                        ModSV=None,
                        alphaem_running=False,
                    )
                    sc2 = Couplings.from_dict(setup)
                    assert sc2.q2_ref == scale_ref
                    assert sc2.a_ref[0] == couplings_ref[0] / 4.0 / np.pi
                    assert sc2.a_ref[1] == couplings_ref[1] / 4.0 / np.pi

        # errors
        with pytest.raises(ValueError):
            Couplings(
                [0, couplings_ref[1]],
                scale_ref,
                threshold_holder.area_walls[1:-1],
                (1.0, 1.0, 1.0),
            )
        with pytest.raises(NotImplementedError):
            Couplings(
                [couplings_ref[0], couplings_ref[1]],
                scale_ref,
                threshold_holder.area_walls[1:-1],
                (1.0, 1.0, 1.0),
                (0, 2),
            )
        with pytest.raises(ValueError):
            Couplings(
                [couplings_ref[0], 0],
                scale_ref,
                threshold_holder.area_walls[1:-1],
                (1.0, 1.0, 1.0),
            )
        with pytest.raises(ValueError):
            Couplings(
                couplings_ref, 0, threshold_holder.area_walls[1:-1], (1.0, 1.0, 1.0)
            )
        with pytest.raises(NotImplementedError):
            Couplings(
                couplings_ref,
                scale_ref,
                threshold_holder.area_walls[1:-1],
                (1.0, 1.0, 1.0),
                (6, 0),
            )
        with pytest.raises(NotImplementedError):
            Couplings(
                couplings_ref,
                scale_ref,
                threshold_holder.area_walls[1:-1],
                (1.0, 1.0, 1.0),
                (1, 3),
            )
        with pytest.raises(ValueError):
            Couplings(
                couplings_ref,
                scale_ref,
                threshold_holder.area_walls[1:-1],
                (1.0, 1.0, 1.0),
                method="ODE",
            )
        with pytest.raises(ValueError):
            Couplings.from_dict(
                dict(
                    alphas=alphas_ref,
                    alphaem=alphaem_ref,
                    Qref=np.sqrt(scale_ref),
                    nfref=None,
                    order=(1, 0),
                    ModEv="FAIL",
                ),
            )
        with pytest.raises(ValueError):
            Couplings.from_dict(
                dict(
                    alphas=alphas_ref,
                    alphaem=alphaem_ref,
                    Qref=np.sqrt(scale_ref),
                    nfref=None,
                    order=(1, 0),
                    ModEv="EXA",
                    HQ="FAIL",
                ),
            )

    def test_ref(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        scale_ref = 91.0**2
        for thresh_setup in thresh_setups:
            for order_s in [1, 2, 3, 4]:
                for order_em in [0, 1, 2]:
                    for method in ["exact", "expanded"]:
                        for alphaem_running in [True, False]:
                            # if order_em == 1 and method == "expanded" and order_s != 0:
                            #    continue
                            # create
                            sc = Couplings(
                                np.array([alphas_ref, alphaem_ref]),
                                scale_ref,
                                thresh_setup,
                                (1.0, 1.0, 1.0),
                                (order_s, order_em),
                                method,
                                alphaem_running=alphaem_running,
                            )
                            np.testing.assert_approx_equal(
                                sc.a(scale_ref)[0], alphas_ref / 4.0 / np.pi
                            )
                            np.testing.assert_approx_equal(
                                sc.a(scale_ref)[1], alphaem_ref / 4.0 / np.pi
                            )

    def test_ref_copy_e841b0dfdee2f31d9ccc1ecee4d9d1a6f6624313(self):
        """Reference settings are an array with QED, but were not copied."""
        thresh_setup = (2.0, 4.75**2.0, 175.0**2.0)
        alphas_ref = 0.35
        alphaem_ref = 0.00781
        scale_ref = 2.0
        couplings_ref = np.array([alphas_ref, alphaem_ref])
        sc = Couplings(
            couplings_ref,
            scale_ref,
            thresh_setup,
            (1.0, 1.0, 1.0),
            (4, 0),
            nf_ref=3,  # reference nf is needed to force the matching
        )
        np.testing.assert_allclose(couplings_ref, np.array([alphas_ref, alphaem_ref]))
        np.testing.assert_allclose(
            sc.a_ref, np.array([alphas_ref, alphaem_ref]) / (4.0 * np.pi)
        )
        # force matching
        sc.a_s(2.0, nf_to=4)
        # of course the object should not have changed!
        np.testing.assert_allclose(couplings_ref, np.array([alphas_ref, alphaem_ref]))
        np.testing.assert_allclose(
            sc.a_ref, np.array([alphas_ref, alphaem_ref]) / (4.0 * np.pi)
        )

    def test_exact_LO(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        scale_ref = 91.0**2
        for thresh_setup in thresh_setups:
            for alphaem_running in [True, False]:
                # in LO expanded  = exact
                sc_expanded = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (1, 0),
                    "expanded",
                    alphaem_running=alphaem_running,
                )
                sc_exact = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (1, 0),
                    "exact",
                    alphaem_running=alphaem_running,
                )
                for q2 in [1, 1e1, 1e2, 1e3, 1e4]:
                    np.testing.assert_allclose(
                        sc_expanded.a(q2)[0], sc_exact.a(q2)[0], rtol=5e-4
                    )
                    np.testing.assert_allclose(
                        sc_expanded.a(q2)[1], sc_exact.a(q2)[1], rtol=5e-4
                    )

    def test_exact_NLO(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        scale_ref = 91.0**2
        for thresh_setup in thresh_setups:
            for alphaem_running in [True, False]:
                # in LO expanded  = exact
                sc_expanded = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (2, 0),
                    "expanded",
                    alphaem_running=alphaem_running,
                )
                sc_exact = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (2, 0),
                    "exact",
                    alphaem_running=alphaem_running,
                )
                for q2 in [1e2, 1e3, 1e4]:
                    np.testing.assert_allclose(
                        sc_expanded.a(q2)[0], sc_exact.a(q2)[0], atol=5e-4
                    )
                    np.testing.assert_allclose(
                        sc_expanded.a(q2)[1], sc_exact.a(q2)[1], atol=5e-4
                    )

    def test_exact_LO_QED(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        scale_ref = 91.0**2
        for PTOs in range(1, 2 + 1):
            for thresh_setup in thresh_setups:
                for alphaem_running in [True, False]:
                    # in LO expanded  = exact
                    sc_expanded = Couplings(
                        np.array([alphas_ref, alphaem_ref]),
                        scale_ref,
                        thresh_setup,
                        (1.0, 1.0, 1.0),
                        (PTOs, 1),
                        "expanded",
                        alphaem_running=alphaem_running,
                    )
                    sc_exact = Couplings(
                        np.array([alphas_ref, alphaem_ref]),
                        scale_ref,
                        thresh_setup,
                        (1.0, 1.0, 1.0),
                        (PTOs, 1),
                        "exact",
                        alphaem_running=alphaem_running,
                    )
                    for q2 in [1e2, 1e3, 1e4]:
                        np.testing.assert_allclose(
                            sc_expanded.a(q2)[0], sc_exact.a(q2)[0], atol=1e-4
                        )
                        np.testing.assert_allclose(
                            sc_expanded.a(q2)[1], sc_exact.a(q2)[1], atol=5e-4
                        )

    def test_exact_NLO_QED(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        scale_ref = 91.0**2
        for PTOs in range(1, 4 + 1):
            for thresh_setup in thresh_setups:
                for alphaem_running in [True, False]:
                    # in LO expanded  = exact
                    sc_expanded = Couplings(
                        np.array([alphas_ref, alphaem_ref]),
                        scale_ref,
                        thresh_setup,
                        (1.0, 1.0, 1.0),
                        (PTOs, 2),
                        "expanded",
                        alphaem_running=alphaem_running,
                    )
                    sc_exact = Couplings(
                        np.array([alphas_ref, alphaem_ref]),
                        scale_ref,
                        thresh_setup,
                        (1.0, 1.0, 1.0),
                        (PTOs, 2),
                        "exact",
                        alphaem_running=alphaem_running,
                    )
                    for q2 in [1e2, 1e3, 1e4]:
                        np.testing.assert_allclose(
                            sc_expanded.a(q2)[0], sc_exact.a(q2)[0], atol=5e-4
                        )
                        np.testing.assert_allclose(
                            sc_expanded.a(q2)[1], sc_exact.a(q2)[1], atol=5e-4
                        )

    def test_exact_NLO_mix(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        scale_ref = 91.0**2
        for thresh_setup in thresh_setups:
            for alphaem_running in [True, False]:
                # in LO expanded  = exact
                sc_expanded = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (2, 2),
                    "expanded",
                    alphaem_running=alphaem_running,
                )
                sc_exact = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (2, 2),
                    "exact",
                    alphaem_running=alphaem_running,
                )
                for q2 in [1e2, 1e3, 1e4]:
                    np.testing.assert_allclose(
                        sc_expanded.a(q2)[0], sc_exact.a(q2)[0], atol=5e-4
                    )
                    np.testing.assert_allclose(
                        sc_expanded.a(q2)[1], sc_exact.a(q2)[1], atol=5e-4
                    )

    def test_exact_N2LO_mix(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        scale_ref = 91.0**2
        for thresh_setup in thresh_setups:
            for alphaem_running in [True, False]:
                # in LO expanded  = exact
                sc_expanded = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (3, 2),
                    "expanded",
                    alphaem_running=alphaem_running,
                )
                sc_exact = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (3, 2),
                    "exact",
                    alphaem_running=alphaem_running,
                )
                for q2 in [1e1, 1e2, 1e3, 1e4]:
                    np.testing.assert_allclose(
                        sc_expanded.a(q2)[0], sc_exact.a(q2)[0], atol=5e-4
                    )
                    np.testing.assert_allclose(
                        sc_expanded.a(q2)[1], sc_exact.a(q2)[1], atol=5e-4
                    )

    def test_exact_N3LO_mix(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        scale_ref = 91.0**2
        for thresh_setup in thresh_setups:
            for alphaem_running in [True, False]:
                # in LO expanded  = exact
                sc_expanded = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (4, 2),
                    "expanded",
                    alphaem_running=alphaem_running,
                )
                sc_exact = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (4, 2),
                    "exact",
                    alphaem_running=alphaem_running,
                )
                for q2 in [1e1, 1e2, 1e3, 1e4]:
                    np.testing.assert_allclose(
                        sc_expanded.a(q2)[0], sc_exact.a(q2)[0], atol=5e-3
                    )
                    np.testing.assert_allclose(
                        sc_expanded.a(q2)[1], sc_exact.a(q2)[1], atol=5e-4
                    )

    def benchmark_expanded_n3lo(self):
        """test N3LO - NNLO expansion with some reference value from Mathematica"""
        Q2 = 100**2
        # use a big alpha_s to enlarge the difference
        alphas_ref = 0.9
        alphaem_ref = 0.00781
        scale_ref = 90**2
        m2c = 2
        m2b = 25
        m2t = 30625
        threshold_list = [m2c, m2b, m2t]
        mathematica_val = -0.000169117
        # collect my values
        as_NNLO = Couplings(
            np.array([alphas_ref, alphaem_ref]),
            scale_ref,
            threshold_list,
            (1.0, 1.0, 1.0),
            order=(3, 0),
            method="expanded",
        )
        as_N3LO = Couplings(
            np.array([alphas_ref, alphaem_ref]),
            scale_ref,
            threshold_list,
            (1.0, 1.0, 1.0),
            order=(4, 0),
            method="expanded",
        )
        np.testing.assert_allclose(
            mathematica_val, as_N3LO.a(Q2)[0] - as_NNLO.a(Q2)[0], rtol=3e-6
        )

    def test_running_alpha(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        scale_ref = 91.0**2
        for thresh_setup in thresh_setups:
            for qcd in range(1, 4 + 1):
                sc_expanded_running = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (qcd, 0),
                    "expanded",
                    alphaem_running=True,
                )
                sc_expanded_fixed = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (qcd, 0),
                    "expanded",
                    alphaem_running=False,
                )
                sc_exact_running = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (qcd, 0),
                    "exact",
                    alphaem_running=True,
                )
                sc_exact_fixed = Couplings(
                    np.array([alphas_ref, alphaem_ref]),
                    scale_ref,
                    thresh_setup,
                    (1.0, 1.0, 1.0),
                    (qcd, 0),
                    "exact",
                    alphaem_running=False,
                )
                for q2 in [1e1, 1e2, 1e3, 1e4]:
                    # for pure qcd running or fixed alpha must be equal
                    np.testing.assert_allclose(
                        sc_expanded_running.a(q2), sc_expanded_fixed.a(q2), rtol=1e-10
                    )
                    np.testing.assert_allclose(
                        sc_exact_running.a(q2), sc_exact_fixed.a(q2), rtol=1e-10
                    )

    def test_compute_aem_as(self):
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        scale_ref = 91.2**2
        thresh_setup = (2, 4, 175)
        scale_target = [5, 10, 50, 100, 150]  # nf = 5
        for alphaem_running in [False]:
            for qcd in range(1, 4 + 1):
                for qed in range(0, 2 + 1):
                    couplings = Couplings(
                        np.array([alphas_ref, alphaem_ref]),
                        scale_ref,
                        thresh_setup,
                        (1.0, 1.0, 1.0),
                        (qcd, qed),
                        "exact",
                        nf_ref=5,
                        alphaem_running=alphaem_running,
                    )
                    a_values = []
                    for Qf in scale_target:
                        a_values.append(couplings.a(Qf**2))
                    for a in a_values:
                        aem = couplings.compute_aem_as(
                            alphaem_ref / 4 / np.pi, alphas_ref / 4 / np.pi, a[0], nf=5
                        )
                        np.testing.assert_allclose(aem, a[1], atol=1e-10, rtol=1e-10)
        for alphaem_running in [True, False]:
            for qcd in range(1, 4 + 1):
                for qed in range(0, 0 + 1):
                    couplings = Couplings(
                        np.array([alphas_ref, alphaem_ref]),
                        scale_ref,
                        thresh_setup,
                        (1.0, 1.0, 1.0),
                        (qcd, qed),
                        "exact",
                        nf_ref=5,
                        alphaem_running=alphaem_running,
                    )
                    a_values = []
                    for Qf in scale_target:
                        a_values.append(couplings.a(Qf**2))
                    for a in a_values:
                        aem = couplings.compute_aem_as(
                            alphaem_ref / 4 / np.pi, alphas_ref / 4 / np.pi, a[0], nf=5
                        )
                        np.testing.assert_allclose(aem, a[1], atol=1e-10, rtol=1e-10)
        for alphaem_running in [True]:
            for qcd in range(1, 4 + 1):
                for qed in range(1, 2 + 1):
                    couplings = Couplings(
                        np.array([alphas_ref, alphaem_ref]),
                        scale_ref,
                        thresh_setup,
                        (1.0, 1.0, 1.0),
                        (qcd, qed),
                        "exact",
                        nf_ref=5,
                        alphaem_running=alphaem_running,
                    )
                    a_values = []
                    for Qf in scale_target:
                        a_values.append(couplings.a(Qf**2))
                    for a in a_values:
                        aem = couplings.compute_aem_as(
                            alphaem_ref / 4 / np.pi, alphas_ref / 4 / np.pi, a[0], nf=5
                        )
                        np.testing.assert_allclose(aem, a[1], atol=1e-6, rtol=1e-2)
        scale_target = [2.1, 2.5, 3.0, 3.5]  # nf = 4
        for alphaem_running in [True]:
            for qcd in range(1, 4 + 1):
                for qed in range(1, 2 + 1):
                    couplings = Couplings(
                        np.array([alphas_ref, alphaem_ref]),
                        scale_ref,
                        thresh_setup,
                        (1.0, 1.0, 1.0),
                        (qcd, qed),
                        "exact",
                        nf_ref=5,
                        alphaem_running=alphaem_running,
                    )
                    a_ref = couplings.a(4.0**2, nf_to=4)
                    a_values = []
                    for Qf in scale_target:
                        a_values.append(couplings.a(Qf**2))
                    for a in a_values:
                        aem = couplings.compute_aem_as(a_ref[1], a_ref[0], a[0], nf=4)
                        np.testing.assert_allclose(aem, a[1], atol=1e-6, rtol=1e-4)

    def test_as_aem(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        alphaem_ref = 0.00781
        scale_ref = 91.0**2
        for thresh_setup in thresh_setups:
            for qcd in range(1, 4 + 1):
                for qed in range(2 + 1):
                    for mode_ev in ["expanded", "exact"]:
                        for q2 in [
                            1.0**2,
                            3.0**2,
                            4.0**2,
                            10.0**2,
                            50**2,
                            100**2,
                            150.0**2,
                            180.0**2,
                        ]:
                            for fact_scale_ratio in [0.5**2, 1.0**2, 2.0**2]:
                                for alphaem_running in [True, False]:
                                    coupling = Couplings(
                                        np.array([alphas_ref, alphaem_ref]),
                                        scale_ref,
                                        thresh_setup,
                                        (1.0, 1.0, 1.0),
                                        (qcd, qed),
                                        mode_ev,
                                        alphaem_running=alphaem_running,
                                    )
                                    np.testing.assert_allclose(
                                        coupling.a(
                                            q2, fact_scale=q2 * fact_scale_ratio
                                        )[0],
                                        coupling.a_s(
                                            q2, fact_scale=q2 * fact_scale_ratio
                                        ),
                                        rtol=1e-10,
                                    )
                                    np.testing.assert_allclose(
                                        coupling.a(
                                            q2, fact_scale=q2 * fact_scale_ratio
                                        )[1],
                                        coupling.a_em(
                                            q2, fact_scale=q2 * fact_scale_ratio
                                        ),
                                        rtol=1e-10,
                                    )
