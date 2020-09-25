# -*- coding: utf-8 -*-
"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.
"""
import numpy as np
import pytest

from eko.strong_coupling import beta_0, beta_1, beta_2, StrongCoupling
from eko import thresholds

# from eko.constants import Constants


class TestBetaFunction:
    def _flav_test(self, function):
        """Check that the given beta function `function` is valid
        for any number of flavours up to 5"""
        for nf in range(5):
            result = function(nf)
            assert result > 0.0

    def _check_result(self, function, NF, value):
        """Check that function evaluated in nf=5
        returns the value `value`"""
        result = function(NF)
        np.testing.assert_approx_equal(result, value, significant=5)

    def test_beta_0(self):
        """Test first beta function coefficient"""
        self._flav_test(beta_0)
        # from hep-ph/9706430
        self._check_result(beta_0, 5, 4 * 23 / 12)

    def test_beta_1(self):
        """Test second beta function coefficient"""
        self._flav_test(beta_1)
        # from hep-ph/9706430
        self._check_result(beta_1, 5, 4 ** 2 * 29 / 12)

    def test_beta_2(self):
        """Test third beta function coefficient"""
        self._flav_test(beta_2)
        # from hep-ph/9706430
        self._check_result(beta_2, 5, 4 ** 3 * 9769 / 3456)


class TestStrongCoupling:
    def test_init(self):
        # prepare
        alphas_ref = 0.118
        scale_ref = 91.0 ** 2
        nf = 4
        threshold_holder = thresholds.ThresholdsConfig(scale_ref, "FFNS", nf=nf)
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
                sc2 = StrongCoupling.from_dict(setup)
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
            {"FNS": "FFNS", "NfFF": 3},
            {"FNS": "FFNS", "NfFF": 4},
            {"FNS": "ZM-VFNS", "mc": 2, "mb": 4, "mt": 175},
        ]
        alphas_ref = 0.118
        scale_ref = 91.0 ** 2
        for thresh_setup in thresh_setups:
            thresh_setup["Q0"] = 1
            thresholds_conf = thresholds.ThresholdsConfig.from_dict(thresh_setup)
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
            {"FNS": "FFNS", "NfFF": 3},
            {"FNS": "FFNS", "NfFF": 4},
            {"FNS": "ZM-VFNS", "mc": 2, "mb": 4, "mt": 175},
        ]
        alphas_ref = 0.118
        scale_ref = 91.0 ** 2
        for thresh_setup in thresh_setups:
            thresh_setup["Q0"] = 1
            thresholds_conf = thresholds.ThresholdsConfig.from_dict(thresh_setup)
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
