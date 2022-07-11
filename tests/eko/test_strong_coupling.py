# -*- coding: utf-8 -*-
"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.
"""
import numpy as np
import pytest

from eko import thresholds
from eko.strong_coupling import StrongCoupling


class TestStrongCoupling:
    def test_from_dict(self):
        d = {
            "alphas": 0.118,
            "Qref": 91.0,
            "nfref": None,
            "Q0": 1,
            "PTO": 0,
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
        }
        sc = StrongCoupling.from_dict(d)
        assert sc.a_s(d["Qref"] ** 2) == d["alphas"] / (4.0 * np.pi)

    def test_init(self):
        # prepare
        alphas_ref = 0.118
        scale_ref = 91.0**2
        nf = 4
        threshold_holder = thresholds.ThresholdsAtlas.ffns(nf)
        # create
        sc = StrongCoupling(
            alphas_ref, scale_ref, threshold_holder.area_walls[1:-1], (1.0, 1.0, 1.0)
        )
        assert sc.q2_ref == scale_ref
        assert sc.as_ref == alphas_ref / 4.0 / np.pi
        # from theory dict
        for ModEv in ["EXP", "EXA"]:
            for PTO in range(2 + 1):
                setup = dict(
                    alphas=alphas_ref,
                    Qref=np.sqrt(scale_ref),
                    nfref=None,
                    PTO=PTO,
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
                )
                sc2 = StrongCoupling.from_dict(setup)
                assert sc2.q2_ref == scale_ref
                assert sc2.as_ref == alphas_ref / 4.0 / np.pi

        # errors
        with pytest.raises(ValueError):
            StrongCoupling(
                0, scale_ref, threshold_holder.area_walls[1:-1], (1.0, 1.0, 1.0)
            )
        with pytest.raises(ValueError):
            StrongCoupling(
                alphas_ref, 0, threshold_holder.area_walls[1:-1], (1.0, 1.0, 1.0)
            )
        with pytest.raises(NotImplementedError):
            StrongCoupling(
                alphas_ref,
                scale_ref,
                threshold_holder.area_walls[1:-1],
                (1.0, 1.0, 1.0),
                4,
            )
        with pytest.raises(ValueError):
            StrongCoupling(
                alphas_ref,
                scale_ref,
                threshold_holder.area_walls[1:-1],
                (1.0, 1.0, 1.0),
                method="ODE",
            )
        with pytest.raises(ValueError):
            StrongCoupling.from_dict(
                dict(
                    alphas=alphas_ref,
                    Qref=np.sqrt(scale_ref),
                    nfref=None,
                    PTO=0,
                    ModEv="FAIL",
                ),
            )
        with pytest.raises(ValueError):
            StrongCoupling.from_dict(
                dict(
                    alphas=alphas_ref,
                    Qref=np.sqrt(scale_ref),
                    nfref=None,
                    PTO=0,
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
        scale_ref = 91.0**2
        for thresh_setup in thresh_setups:
            for order in [0, 1, 2, 3]:
                for method in ["exact", "expanded"]:
                    # create
                    sc = StrongCoupling(
                        alphas_ref,
                        scale_ref,
                        thresh_setup,
                        (1.0, 1.0, 1.0),
                        order,
                        method,
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
        scale_ref = 91.0**2
        for thresh_setup in thresh_setups:
            # in LO expanded  = exact
            sc_expanded = StrongCoupling(
                alphas_ref, scale_ref, thresh_setup, (1.0, 1.0, 1.0), 0, "expanded"
            )
            sc_exact = StrongCoupling(
                alphas_ref, scale_ref, thresh_setup, (1.0, 1.0, 1.0), 0, "exact"
            )
            for q2 in [1, 1e1, 1e2, 1e3, 1e4]:
                np.testing.assert_allclose(
                    sc_expanded.a_s(q2), sc_exact.a_s(q2), rtol=5e-4
                )

    def benchmark_expanded_n3lo(self):
        """test N3LO - NNLO expansion with some reference value from Mathematica"""
        Q2 = 100**2
        # use a big alpha_s to enlarge the difference
        alphas_ref = 0.9
        scale_ref = 90**2
        m2c = 2
        m2b = 25
        m2t = 30625
        threshold_list = [m2c, m2b, m2t]
        mathematica_val = -0.000169117
        # collect my values
        as_NNLO = StrongCoupling(
            alphas_ref,
            scale_ref,
            threshold_list,
            (1.0, 1.0, 1.0),
            order=2,
            method="expanded",
        )
        as_N3LO = StrongCoupling(
            alphas_ref,
            scale_ref,
            threshold_list,
            (1.0, 1.0, 1.0),
            order=3,
            method="expanded",
        )
        np.testing.assert_allclose(
            mathematica_val, as_N3LO.a_s(Q2) - as_NNLO.a_s(Q2), rtol=3e-6
        )
