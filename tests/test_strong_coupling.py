# -*- coding: utf-8 -*-
"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.
"""
import numpy as np
import pytest

from eko.strong_coupling import StrongCoupling
from eko import thresholds


class TestStrongCoupling:
    def test_init(self):
        # prepare
        alphas_ref = 0.118
        scale_ref = 91.0 ** 2
        nf = 4
        threshold_holder = thresholds.ThresholdsAtlas.ffns(nf)
        # create
        sc = StrongCoupling(alphas_ref, scale_ref, threshold_holder)
        assert sc.q2_ref == scale_ref
        assert sc.as_ref == alphas_ref / 4.0 / np.pi
        # from theory dict
        for ModEv in ["EXP", "EXA"]:
            for PTO in range(2 + 1):
                setup = dict(
                    alphas=alphas_ref,
                    Qref=np.sqrt(scale_ref),
                    PTO=PTO,
                    ModEv=ModEv,
                    FNS="FFNS",
                    NfFF=nf,
                    Q0=2,
                )
                sc2 = StrongCoupling.from_dict(setup, threshold_holder)
                assert sc2.q2_ref == scale_ref
                assert sc2.as_ref == alphas_ref / 4.0 / np.pi

        # errors
        with pytest.raises(ValueError):
            StrongCoupling(0, scale_ref, threshold_holder)
        with pytest.raises(ValueError):
            StrongCoupling(alphas_ref, 0, threshold_holder)
        with pytest.raises(ValueError):
            StrongCoupling(alphas_ref, scale_ref, None)
        with pytest.raises(NotImplementedError):
            StrongCoupling(alphas_ref, scale_ref, threshold_holder, 3)
        with pytest.raises(ValueError):
            StrongCoupling(alphas_ref, scale_ref, threshold_holder, method="ODE")
        with pytest.raises(ValueError):
            StrongCoupling.from_dict(
                dict(alphas=alphas_ref, Qref=np.sqrt(scale_ref), PTO=0, ModEv="FAIL"),
                threshold_holder,
            )

    def test_ref(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        scale_ref = 91.0 ** 2
        for thresh_setup in thresh_setups:
            thresholds_conf = thresholds.ThresholdsAtlas(thresh_setup, 1)
            for order in [0, 1, 2]:
                for method in ["exact", "expanded"]:
                    # create
                    sc = StrongCoupling(
                        alphas_ref, scale_ref, thresholds_conf, order, method
                    )
                    np.testing.assert_approx_equal(
                        sc.a_s(scale_ref), alphas_ref / 4.0 / np.pi
                    )

    def test_exact_LO(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alphas_ref = 0.118
        scale_ref = 91.0 ** 2
        for thresh_setup in thresh_setups:
            thresholds_conf = thresholds.ThresholdsAtlas(thresh_setup, 1)
            # in LO expanded  = exact
            sc_expanded = StrongCoupling(
                alphas_ref, scale_ref, thresholds_conf, 0, "expanded"
            )
            sc_exact = StrongCoupling(
                alphas_ref, scale_ref, thresholds_conf, 0, "exact"
            )
            for q2 in [1, 1e1, 1e2, 1e3, 1e4]:
                np.testing.assert_allclose(
                    sc_expanded.a_s(q2), sc_exact.a_s(q2), rtol=5e-4
                )
