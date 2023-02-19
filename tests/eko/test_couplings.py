"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.
"""
import copy
import enum
from math import nan

import numpy as np
import pytest

from eko.couplings import Couplings, compute_matching_coeffs_up, couplings_mod_ev
from eko.io.types import (
    CouplingEvolutionMethod,
    CouplingsRef,
    EvolutionMethod,
    MatchingScales,
    QuarkMassSchemes,
)

masses = [m**2 for m in (2.0, 4.5, 175.0)]


class FakeEM(enum.Enum):
    BLUB = "blub"


def test_couplings_mod_ev():
    assert (
        couplings_mod_ev(EvolutionMethod.ITERATE_EXACT) == CouplingEvolutionMethod.EXACT
    )
    assert (
        couplings_mod_ev(EvolutionMethod.TRUNCATED) == CouplingEvolutionMethod.EXPANDED
    )
    with pytest.raises(ValueError, match="BLUB"):
        couplings_mod_ev(FakeEM.BLUB)


def test_compute_matching_coeffs_up():
    for mass_scheme in ["MSBAR", "POLE"]:
        for nf in [3, 4, 5]:
            c = compute_matching_coeffs_up(mass_scheme, nf)
            # has to be quasi triangular
            np.testing.assert_allclose(c[0, :], 0.0)
            for k in range(1, 3 + 1):
                np.testing.assert_allclose(
                    c[k, (k + 1) :],
                    0.0,
                    err_msg=f"mass_scheme={mass_scheme},nf={nf},k={k}",
                )


class TestCouplings:
    def test_init(self):
        alpharef = (0.118, 0.00781)
        muref = 91.0
        couplings = CouplingsRef.from_dict(
            dict(
                alphas=[alpharef[0], muref],
                alphaem=[alpharef[1], nan],
                num_flavs_ref=None,
                max_num_flavs=6,
            )
        )
        order = (1, 0)
        evmod = CouplingEvolutionMethod.EXACT
        # create
        sc = Couplings(
            couplings,
            order,
            evmod,
            masses,
            hqm_scheme=QuarkMassSchemes.POLE,
            thresholds_ratios=[1.0, 1.0, 1.0],
        )
        assert sc.q2_ref == muref**2
        assert sc.a_ref[0] == alpharef[0] / 4.0 / np.pi
        assert sc.a(muref**2)[0] == alpharef[0] / (4.0 * np.pi)
        assert sc.a_ref[1] == alpharef[1] / 4.0 / np.pi

        # errors
        with pytest.raises(ValueError):
            coup1 = copy.deepcopy(couplings)
            coup1.alphas.value = 0
            Couplings(
                coup1,
                order,
                evmod,
                masses,
                hqm_scheme=QuarkMassSchemes.POLE,
                thresholds_ratios=[1.0, 1.0, 1.0],
            )
        with pytest.raises(NotImplementedError):
            Couplings(
                couplings,
                (0, 2),
                evmod,
                masses,
                hqm_scheme=QuarkMassSchemes.POLE,
                thresholds_ratios=MatchingScales(c=1.0, b=1.0, t=1.0),
            )
        with pytest.raises(ValueError):
            coup2 = copy.deepcopy(couplings)
            coup2.alphaem.value = 0
            Couplings(
                coup2,
                order,
                evmod,
                masses,
                hqm_scheme=QuarkMassSchemes.POLE,
                thresholds_ratios=[1.0, 1.0, 1.0],
            )
        with pytest.raises(ValueError):
            coup3 = copy.deepcopy(couplings)
            coup3.alphas.scale = 0
            Couplings(
                coup3,
                order,
                evmod,
                masses,
                hqm_scheme=QuarkMassSchemes.POLE,
                thresholds_ratios=[1.0, 1.0, 1.0],
            )
        with pytest.raises(NotImplementedError):
            Couplings(
                couplings,
                (6, 0),
                evmod,
                masses,
                hqm_scheme=QuarkMassSchemes.POLE,
                thresholds_ratios=[1.0, 1.0, 1.0],
            )
        with pytest.raises(NotImplementedError):
            Couplings(
                couplings,
                (1, 3),
                evmod,
                masses,
                hqm_scheme=QuarkMassSchemes.POLE,
                thresholds_ratios=[1.0, 1.0, 1.0],
            )
        with pytest.raises(ValueError):
            Couplings(
                couplings,
                (2, 0),
                FakeEM.BLUB,
                masses,
                hqm_scheme=QuarkMassSchemes.POLE,
                thresholds_ratios=[1.0, 1.0, 1.0],
            )

    def test_ref(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alpharef = (0.118, 0.00781)
        muref = 91.0
        couplings = CouplingsRef.from_dict(
            dict(
                alphas=[alpharef[0], muref],
                alphaem=[alpharef[1], nan],
                num_flavs_ref=None,
                max_num_flavs=6,
            )
        )
        for thresh_setup in thresh_setups:
            for order_s in [1, 2, 3, 4]:
                for order_em in [0, 1, 2]:
                    for evmod in CouplingEvolutionMethod:
                        # if order_em == 1 and method == "expanded" and order_s != 0:
                        #    continue
                        # create
                        sc = Couplings(
                            couplings,
                            (order_s, order_em),
                            evmod,
                            masses,
                            hqm_scheme=QuarkMassSchemes.POLE,
                            thresholds_ratios=thresh_setup,
                        )
                        np.testing.assert_approx_equal(
                            sc.a(muref**2)[0], alpharef[0] / 4.0 / np.pi
                        )
                        np.testing.assert_approx_equal(
                            sc.a(muref**2)[1], alpharef[1] / 4.0 / np.pi
                        )

    def test_ref_copy_e841b0dfdee2f31d9ccc1ecee4d9d1a6f6624313(self):
        """Reference settings are an array with QED, but were not copied."""
        thresh_setup = (1.51, 4.75, 175.0)
        alpharef = np.array([0.35, 0.00781])
        muref = 2.0
        couplings = CouplingsRef.from_dict(
            dict(
                alphas=[alpharef[0], muref],
                alphaem=[alpharef[1], nan],
                num_flavs_ref=3,  # reference nf is needed to force the matching
                max_num_flavs=6,
            )
        )
        sc = Couplings(
            couplings,
            (4, 0),
            method=CouplingEvolutionMethod.EXACT,
            masses=masses,
            hqm_scheme=QuarkMassSchemes.POLE,
            thresholds_ratios=thresh_setup,
        )
        np.testing.assert_allclose(sc.a_ref, np.array(alpharef) / (4.0 * np.pi))
        # force matching
        sc.a_s(2.0, nf_to=4)
        # of course the object should not have changed!
        np.testing.assert_allclose(sc.a_ref, np.array(alpharef) / (4.0 * np.pi))

    def test_exact(self):
        # prepare
        thresh_setups = [
            (np.inf, np.inf, np.inf),
            (0, np.inf, np.inf),
            (2, 4, 175),
        ]
        alpharef = np.array([0.118, 0.00781])
        muref = 91.0
        for thresh_setup in thresh_setups:
            for qcd in range(1, 4 + 1):
                for qed in range(2 + 1):
                    for qedref in [muref, nan]:  # testing both running and fixed alpha
                        pto = (qcd, qed)
                        couplings = CouplingsRef.from_dict(
                            dict(
                                alphas=[alpharef[0], muref],
                                alphaem=[alpharef[1], qedref],
                                num_flavs_ref=None,
                                max_num_flavs=6,
                            )
                        )
                        sc_expanded = Couplings(
                            couplings,
                            pto,
                            method=CouplingEvolutionMethod.EXPANDED,
                            masses=masses,
                            hqm_scheme=QuarkMassSchemes.POLE,
                            thresholds_ratios=thresh_setup,
                        )
                        sc_exact = Couplings(
                            couplings,
                            pto,
                            method=CouplingEvolutionMethod.EXACT,
                            masses=masses,
                            hqm_scheme=QuarkMassSchemes.POLE,
                            thresholds_ratios=thresh_setup,
                        )
                        if pto in [(1, 0), (1, 1), (1, 2)]:
                            precisions = (5e-4, 5e-4)
                        else:
                            precisions = (5e-3, 5e-4)
                        for q2 in [1, 1e1, 1e2, 1e3, 1e4]:
                            # At LO (either QCD or QED LO) the exact and expanded
                            # solutions coincide, while beyond LO they don't.
                            # Therefore if the path is too long they start being different.
                            if q2 in [1, 1e1] and pto not in [(1, 0)]:
                                continue
                            np.testing.assert_allclose(
                                sc_expanded.a(q2)[0],
                                sc_exact.a(q2)[0],
                                rtol=precisions[0],
                            )
                            np.testing.assert_allclose(
                                sc_expanded.a(q2)[1],
                                sc_exact.a(q2)[1],
                                rtol=precisions[1],
                            )
                            if qedref is nan or qed == 0:
                                np.testing.assert_allclose(
                                    sc_expanded.a(q2)[1],
                                    alpharef[1] / (4 * np.pi),
                                    rtol=1e-10,
                                )
                                np.testing.assert_allclose(
                                    sc_exact.a(q2)[1],
                                    alpharef[1] / (4 * np.pi),
                                    rtol=1e-10,
                                )

    def benchmark_expanded_n3lo(self):
        """test N3LO - NNLO expansion with some reference value from Mathematica"""
        Q2 = 100**2
        # use a big alpha_s to enlarge the difference
        alpharef = np.array([0.9, 0.00781])
        muref = 90.0
        couplings = CouplingsRef.from_dict(
            dict(
                alphas=[alpharef[0], muref],
                alphaem=[alpharef[1], nan],
                num_flavs_ref=None,
                max_num_flavs=6,
            )
        )
        m2c = 2
        m2b = 25
        m2t = 30625
        threshold_list = [m2c / masses[0], m2b / masses[1], m2t / masses[2]]
        mathematica_val = -0.000169117
        # collect my values
        as_NNLO = Couplings(
            couplings,
            order=(3, 0),
            method=CouplingEvolutionMethod.EXPANDED,
            masses=masses,
            hqm_scheme=QuarkMassSchemes.POLE,
            thresholds_ratios=threshold_list,
        )
        as_N3LO = Couplings(
            couplings,
            order=(4, 0),
            method=CouplingEvolutionMethod.EXPANDED,
            masses=masses,
            hqm_scheme=QuarkMassSchemes.POLE,
            thresholds_ratios=threshold_list,
        )
        np.testing.assert_allclose(
            mathematica_val, as_N3LO.a(Q2)[0] - as_NNLO.a(Q2)[0], rtol=5e-7
        )

    # def test_compute_aem_as(self):
    #     alphas_ref = 0.118
    #     alphaem_ref = 0.00781
    #     scale_ref = 91.2**2
    #     thresh_setup = (2, 4, 175)
    #     scale_target = [5, 10, 50, 100, 150]  # nf = 5
    #     for alphaem_running in [False]:
    #         for qcd in range(1, 4 + 1):
    #             for qed in range(0, 2 + 1):
    #                 couplings = Couplings(
    #                     np.array([alphas_ref, alphaem_ref]),
    #                     scale_ref,
    #                     thresh_setup,
    #                     (1.0, 1.0, 1.0),
    #                     (qcd, qed),
    #                     "exact",
    #                     nf_ref=5,
    #                     alphaem_running=alphaem_running,
    #                 )
    #                 a_values = []
    #                 for Qf in scale_target:
    #                     a_values.append(couplings.a(Qf**2))
    #                 for a in a_values:
    #                     aem = couplings.compute_aem_as(
    #                         alphaem_ref / 4 / np.pi, alphas_ref / 4 / np.pi, a[0], nf=5
    #                     )
    #                     np.testing.assert_allclose(aem, a[1], atol=1e-10, rtol=1e-10)
    #     for alphaem_running in [True, False]:
    #         for qcd in range(1, 4 + 1):
    #             for qed in range(0, 0 + 1):
    #                 couplings = Couplings(
    #                     np.array([alphas_ref, alphaem_ref]),
    #                     scale_ref,
    #                     thresh_setup,
    #                     (1.0, 1.0, 1.0),
    #                     (qcd, qed),
    #                     "exact",
    #                     nf_ref=5,
    #                     alphaem_running=alphaem_running,
    #                 )
    #                 a_values = []
    #                 for Qf in scale_target:
    #                     a_values.append(couplings.a(Qf**2))
    #                 for a in a_values:
    #                     aem = couplings.compute_aem_as(
    #                         alphaem_ref / 4 / np.pi, alphas_ref / 4 / np.pi, a[0], nf=5
    #                     )
    #                     np.testing.assert_allclose(aem, a[1], atol=1e-10, rtol=1e-10)
    #     for alphaem_running in [True]:
    #         for qcd in range(1, 4 + 1):
    #             for qed in range(1, 2 + 1):
    #                 couplings = Couplings(
    #                     np.array([alphas_ref, alphaem_ref]),
    #                     scale_ref,
    #                     thresh_setup,
    #                     (1.0, 1.0, 1.0),
    #                     (qcd, qed),
    #                     "exact",
    #                     nf_ref=5,
    #                     alphaem_running=alphaem_running,
    #                 )
    #                 a_values = []
    #                 for Qf in scale_target:
    #                     a_values.append(couplings.a(Qf**2))
    #                 for a in a_values:
    #                     aem = couplings.compute_aem_as(
    #                         alphaem_ref / 4 / np.pi, alphas_ref / 4 / np.pi, a[0], nf=5
    #                     )
    #                     np.testing.assert_allclose(aem, a[1], atol=1e-6, rtol=1e-2)
    #     scale_target = [2.1, 2.5, 3.0, 3.5]  # nf = 4
    #     for alphaem_running in [True]:
    #         for qcd in range(1, 4 + 1):
    #             for qed in range(1, 2 + 1):
    #                 couplings = Couplings(
    #                     np.array([alphas_ref, alphaem_ref]),
    #                     scale_ref,
    #                     thresh_setup,
    #                     (1.0, 1.0, 1.0),
    #                     (qcd, qed),
    #                     "exact",
    #                     nf_ref=5,
    #                     alphaem_running=alphaem_running,
    #                 )
    #                 a_ref = couplings.a(4.0**2, nf_to=4)
    #                 a_values = []
    #                 for Qf in scale_target:
    #                     a_values.append(couplings.a(Qf**2))
    #                 for a in a_values:
    #                     aem = couplings.compute_aem_as(a_ref[1], a_ref[0], a[0], nf=4)
    #                     np.testing.assert_allclose(aem, a[1], atol=1e-6, rtol=1e-4)

    # def test_as_aem(self):
    #     # prepare
    #     thresh_setups = [
    #         (np.inf, np.inf, np.inf),
    #         (0, np.inf, np.inf),
    #         (2, 4, 175),
    #     ]
    #     alphas_ref = 0.118
    #     alphaem_ref = 0.00781
    #     scale_ref = 91.0**2
    #     for thresh_setup in thresh_setups:
    #         for qcd in range(1, 4 + 1):
    #             for qed in range(2 + 1):
    #                 for mode_ev in ["expanded", "exact"]:
    #                     for q2 in [
    #                         1.0**2,
    #                         3.0**2,
    #                         4.0**2,
    #                         10.0**2,
    #                         50**2,
    #                         100**2,
    #                         150.0**2,
    #                         180.0**2,
    #                     ]:
    #                         for fact_scale_ratio in [0.5**2, 1.0**2, 2.0**2]:
    #                             for alphaem_running in [True, False]:
    #                                 coupling = Couplings(
    #                                     np.array([alphas_ref, alphaem_ref]),
    #                                     scale_ref,
    #                                     thresh_setup,
    #                                     (1.0, 1.0, 1.0),
    #                                     (qcd, qed),
    #                                     mode_ev,
    #                                     alphaem_running=alphaem_running,
    #                                 )
    #                                 np.testing.assert_allclose(
    #                                     coupling.a(
    #                                         q2, fact_scale=q2 * fact_scale_ratio
    #                                     )[0],
    #                                     coupling.a_s(
    #                                         q2, fact_scale=q2 * fact_scale_ratio
    #                                     ),
    #                                     rtol=1e-10,
    #                                 )
    #                                 np.testing.assert_allclose(
    #                                     coupling.a(
    #                                         q2, fact_scale=q2 * fact_scale_ratio
    #                                     )[1],
    #                                     coupling.a_em(
    #                                         q2, fact_scale=q2 * fact_scale_ratio
    #                                     ),
    #                                     rtol=1e-10,
    #                                 )
