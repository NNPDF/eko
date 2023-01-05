"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.
"""
import copy
from math import nan

import numpy as np
import pytest

from eko.couplings import Couplings
from eko.io.types import (
    CouplingEvolutionMethod,
    CouplingsRef,
    MatchingScales,
    QuarkMassSchemes,
)

masses = [m**2 for m in (2.0, 4.5, 175.0)]


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
            thresholds_ratios=MatchingScales(c=1.0, b=1.0, t=1.0),
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
                thresholds_ratios=MatchingScales(c=1.0, b=1.0, t=1.0),
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
                thresholds_ratios=MatchingScales(c=1.0, b=1.0, t=1.0),
            )
        with pytest.raises(NotImplementedError):
            Couplings(
                couplings,
                (6, 0),
                evmod,
                masses,
                hqm_scheme=QuarkMassSchemes.POLE,
                thresholds_ratios=MatchingScales(c=1.0, b=1.0, t=1.0),
            )
        with pytest.raises(NotImplementedError):
            Couplings(
                couplings,
                (1, 3),
                evmod,
                masses,
                hqm_scheme=QuarkMassSchemes.POLE,
                thresholds_ratios=MatchingScales(c=1.0, b=1.0, t=1.0),
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
            threshs = MatchingScales.from_dict(dict(zip("cbt", thresh_setup)))
            for order_s in [0, 1, 2, 3, 4]:
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
                            thresholds_ratios=threshs,
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
        threshs = MatchingScales.from_dict(dict(zip("cbt", thresh_setup)))
        sc = Couplings(
            couplings,
            (4, 0),
            method=CouplingEvolutionMethod.EXACT,
            masses=masses,
            hqm_scheme=QuarkMassSchemes.POLE,
            thresholds_ratios=threshs,
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
        couplings = CouplingsRef.from_dict(
            dict(
                alphas=[alpharef[0], muref],
                alphaem=[alpharef[1], muref],
                num_flavs_ref=None,
                max_num_flavs=6,
            )
        )
        for thresh_setup in thresh_setups:
            for qcd in range(1, 4 + 1):
                for qed in range(2 + 1):
                    pto = (qcd, qed)
                    threshs = MatchingScales.from_dict(dict(zip("cbt", thresh_setup)))
                    sc_expanded = Couplings(
                        couplings,
                        pto,
                        method=CouplingEvolutionMethod.EXPANDED,
                        masses=masses,
                        hqm_scheme=QuarkMassSchemes.POLE,
                        thresholds_ratios=threshs,
                    )
                    sc_exact = Couplings(
                        couplings,
                        pto,
                        method=CouplingEvolutionMethod.EXACT,
                        masses=masses,
                        hqm_scheme=QuarkMassSchemes.POLE,
                        thresholds_ratios=threshs,
                    )
                    if pto in [(1, 0), (1, 1), (1, 2)]:
                        precisions = (5e-4, 5e-5)
                    else:
                        precisions = (5e-3, 5e-4)
                    for q2 in [1, 1e1, 1e2, 1e3, 1e4]:
                        # At LO (either QCD or QED LO) the exact and expanded
                        # solutions coincide, while beyond LO they don't.
                        # Therefore if the path is too long they start being different.
                        if q2 in [1, 1e1] and pto not in [(1, 0), (0, 1)]:
                            continue
                        np.testing.assert_allclose(
                            sc_expanded.a(q2)[0], sc_exact.a(q2)[0], rtol=precisions[0]
                        )
                        np.testing.assert_allclose(
                            sc_expanded.a(q2)[1], sc_exact.a(q2)[1], rtol=precisions[1]
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
        threshold_list = [m2c, m2b, m2t]
        threshs = MatchingScales.from_dict(dict(zip("cbt", threshold_list)))
        mathematica_val = -0.000169117
        # collect my values
        as_NNLO = Couplings(
            couplings,
            order=(3, 0),
            method=CouplingEvolutionMethod.EXPANDED,
            masses=masses,
            hqm_scheme=QuarkMassSchemes.POLE,
            thresholds_ratios=threshs,
        )
        as_N3LO = Couplings(
            couplings,
            order=(4, 0),
            method=CouplingEvolutionMethod.EXPANDED,
            masses=masses,
            hqm_scheme=QuarkMassSchemes.POLE,
            thresholds_ratios=threshs,
        )
        np.testing.assert_allclose(
            mathematica_val, as_N3LO.a(Q2)[0] - as_NNLO.a(Q2)[0], rtol=3e-6
        )
