# -*- coding: utf-8 -*-
"""This module benchmarks alpha_s against LHAPDF and APFEL."""
import numpy as np

from eko.beta import beta
from eko.strong_coupling import StrongCoupling
from eko import thresholds

# try to load LHAPDF - if not available, we'll use the cached values
try:
    import lhapdf

    use_LHAPDF = True
except ImportError:
    use_LHAPDF = False

# try to load APFEL - if not available, we'll use the cached values
try:
    import apfel

    use_APFEL = True
except ImportError:
    use_APFEL = False


class BenchmarkStrongCoupling:
    def test_a_s(self):
        """Tests the value of alpha_s (for now only at LO)
        for a given set of parameters
        """
        # TODO @JCM: we need a source for this!
        known_val = 0.0091807954
        ref_alpha_s = 0.1181
        ref_mu2 = 90
        ask_q2 = 125
        threshold_holder = thresholds.ThresholdsAtlas([0, 0, np.inf], ref_mu2)
        as_FFNS_LO = StrongCoupling(ref_alpha_s, ref_mu2, threshold_holder, order=0)
        # check local
        np.testing.assert_approx_equal(
            as_FFNS_LO.a_s(ref_mu2), ref_alpha_s / 4.0 / np.pi
        )
        # check high
        result = as_FFNS_LO.a_s(ask_q2)
        np.testing.assert_approx_equal(result, known_val, significant=7)

    def benchmark_LHA_paper(self):
        """Check to :cite:`Giele:2002hx` and :cite:`Dittmar:2005ed`"""
        # LO - FFNS
        # note that the LO-FFNS value reported in :cite:`Giele:2002hx`
        # was corrected in :cite:`Dittmar:2005ed`
        threshold_holder = thresholds.ThresholdsAtlas((0, np.inf, np.inf))
        as_FFNS_LO = StrongCoupling(0.35, 2, threshold_holder, order=0)
        me = as_FFNS_LO.a_s(1e4) * 4 * np.pi
        ref = 0.117574
        np.testing.assert_approx_equal(me, ref, significant=6)
        # LO - VFNS
        threshold_list = [2, pow(4.5, 2), pow(175, 2)]
        threshold_holder = thresholds.ThresholdsAtlas(threshold_list)
        as_VFNS_LO = StrongCoupling(0.35, 2, threshold_holder, order=0)
        me = as_VFNS_LO.a_s(1e4) * 4 * np.pi
        ref = 0.122306
        np.testing.assert_approx_equal(me, ref, significant=6)

    def benchmark_APFEL_ffns(self):
        Q2s = [1e1, 1e2, 1e3, 1e4]
        alphas_ref = 0.118
        scale_ref = 91.0 ** 2
        nf = 4
        apfel_vals_dict = {
            0: np.array(
                [
                    0.01980124164841284,
                    0.014349241933308077,
                    0.01125134009147229,
                    0.009253560532725833,
                ]
            ),
            1: np.array(
                [
                    0.021603236069744878,
                    0.014887068327193985,
                    0.01139235082295531,
                    0.009245832041857378,
                ]
            ),
            2: np.array(
                [
                    0.02204088975651748,
                    0.014966690529271008,
                    0.011406500825908607,
                    0.009245271638953058,
                ]
            ),
        }
        # collect my values
        threshold_holder = thresholds.ThresholdsAtlas.ffns(nf)
        for order in [0, 1, 2]:
            as_FFNS = StrongCoupling(
                alphas_ref,
                scale_ref,
                threshold_holder,
                order=order,
                method="expanded",
            )
            my_vals = []
            for Q2 in Q2s:
                my_vals.append(as_FFNS.a_s(Q2))
            # get APFEL numbers - if available else use cache
            apfel_vals = apfel_vals_dict[order]
            if use_APFEL:
                # run apfel
                apfel.CleanUp()
                apfel.SetTheory("QCD")
                apfel.SetPerturbativeOrder(order)
                apfel.SetAlphaEvolution("expanded")
                apfel.SetAlphaQCDRef(alphas_ref, np.sqrt(scale_ref))
                apfel.SetFFNS(nf)
                apfel.SetRenFacRatio(1)
                # collect a_s
                apfel_vals_cur = []
                for Q2 in Q2s:
                    apfel_vals_cur.append(apfel.AlphaQCD(np.sqrt(Q2)) / (4.0 * np.pi))
                # print(apfel_vals_cur)
                np.testing.assert_allclose(apfel_vals, np.array(apfel_vals_cur))
            # check myself to APFEL
            np.testing.assert_allclose(apfel_vals, np.array(my_vals))

    def benchmark_APFEL_vfns(self):
        Q2s = [1, 2 ** 2, 3 ** 2, 90 ** 2, 100 ** 2]
        alphas_ref = 0.118
        scale_ref = 91.0 ** 2
        threshold_list = np.power([2, 4, 175], 2)
        apfel_vals_dict = {
            0: np.array(
                [
                    0.028938898786215545,
                    0.021262022520127353,
                    0.018590827846469413,
                    0.009405104970805002,
                    0.00926434063784546,
                ]
            ),
            1: np.array(
                [
                    0.035670881093047654,
                    0.02337584106433519,
                    0.01985110421500437,
                    0.009405815313164215,
                    0.009258502199861199,
                ]
            ),
            2: np.array(
                [
                    0.03745593700854872,
                    0.023692463391822537,
                    0.019999870769373283,
                    0.009405846627291407,
                    0.009258253034683823,
                ]
            ),
        }
        # collect my values
        threshold_holder = thresholds.ThresholdsAtlas(threshold_list)
        for order in [0, 1, 2]:
            as_VFNS = StrongCoupling(
                alphas_ref,
                scale_ref,
                threshold_holder,
                order=order,
                method="expanded",
            )
            my_vals = []
            for Q2 in Q2s:
                my_vals.append(as_VFNS.a_s(Q2))
            # get APFEL numbers - if available else use cache
            apfel_vals = apfel_vals_dict[order]
            if use_APFEL:
                # run apfel
                apfel.CleanUp()
                apfel.SetTheory("QCD")
                apfel.SetPerturbativeOrder(order)
                apfel.SetAlphaEvolution("expanded")
                apfel.SetAlphaQCDRef(alphas_ref, np.sqrt(scale_ref))
                apfel.SetVFNS()
                apfel.SetPoleMasses(*np.sqrt(threshold_list))
                apfel.SetRenFacRatio(1)
                apfel.InitializeAPFEL()
                # collect a_s
                apfel_vals_cur = []
                for Q2 in Q2s:
                    apfel_vals_cur.append(apfel.AlphaQCD(np.sqrt(Q2)) / (4.0 * np.pi))
                # print(apfel_vals_cur)
                np.testing.assert_allclose(apfel_vals, np.array(apfel_vals_cur))
            # check myself to APFEL
            np.testing.assert_allclose(apfel_vals, np.array(my_vals))

    def _get_Lambda2_LO(self, as_ref, scale_ref, nf):
        """Transformation to Lambda_QCD"""
        beta0 = beta(0, nf)
        return scale_ref * np.exp(-1.0 / (as_ref * beta0))

    def benchmark_lhapdf_ffns_lo(self):
        """test FFNS LO towards LHAPDF"""
        Q2s = [1, 1e1, 1e2, 1e3, 1e4]
        alphas_ref = 0.118
        scale_ref = 91.0 ** 2
        nf = 4
        # collect my values
        threshold_holder = thresholds.ThresholdsAtlas.ffns(nf)
        as_FFNS_LO = StrongCoupling(alphas_ref, scale_ref, threshold_holder, order=0)
        my_vals = []
        for Q2 in Q2s:
            my_vals.append(as_FFNS_LO.a_s(Q2))
        # get LHAPDF numbers - if available else use cache
        lhapdf_vals = np.array(
            [
                0.031934929816669545,
                0.019801241565290697,
                0.01434924187307247,
                0.01125134004424113,
                0.009253560493881005,
            ]
        )
        if use_LHAPDF:
            # run lhapdf
            as_lhapdf = lhapdf.mkBareAlphaS("analytic")
            as_lhapdf.setOrderQCD(1)
            as_lhapdf.setFlavorScheme("FIXED", nf)
            Lambda2 = self._get_Lambda2_LO(alphas_ref / (4.0 * np.pi), scale_ref, nf)
            as_lhapdf.setLambda(nf, np.sqrt(Lambda2))
            # collect a_s
            lhapdf_vals_cur = []
            for Q2 in Q2s:
                lhapdf_vals_cur.append(as_lhapdf.alphasQ2(Q2) / (4.0 * np.pi))
            # print(lhapdf_vals_cur)
            np.testing.assert_allclose(lhapdf_vals, np.array(lhapdf_vals_cur))
        # check myself to LHAPDF
        np.testing.assert_allclose(lhapdf_vals, np.array(my_vals), rtol=5e-4)

    def benchmark_apfel_exact(self):
        """test exact towards APFEL"""
        Q2s = [1e1, 1e2, 1e3, 1e4]
        alphas_ref = 0.118
        scale_ref = 90 ** 2
        # collect my values
        threshold_holder = thresholds.ThresholdsAtlas.ffns(3)
        # LHAPDF cache
        apfel_vals_dict = {
            0: np.array(
                [
                    0.021635019899707245,
                    0.014937719308417242,
                    0.01140668497649318,
                    0.009225844999427163,
                ]
            ),
            1: np.array(
                [
                    0.025027723843261098,
                    0.015730685887616093,
                    0.01159096381106341,
                    0.009215179564010682,
                ]
            ),
            2: np.array(
                [
                    0.025717091835015565,
                    0.01583723253352162,
                    0.011610857909393214,
                    0.009214183434685514,
                ]
            ),
        }
        for order in range(2 + 1):
            sc = StrongCoupling(
                alphas_ref,
                scale_ref,
                threshold_holder,
                order=order,
                method="exact",
            )
            my_vals = []
            for Q2 in Q2s:
                my_vals.append(sc.a_s(Q2))
            # get APFEL numbers - if available else use cache
            apfel_vals = apfel_vals_dict[order]
            if use_APFEL:
                # run apfel
                apfel.CleanUp()
                apfel.SetTheory("QCD")
                apfel.SetPerturbativeOrder(order)
                apfel.SetAlphaEvolution("exact")
                apfel.SetAlphaQCDRef(alphas_ref, np.sqrt(scale_ref))
                apfel.SetFFNS(3)
                apfel.SetRenFacRatio(1)
                apfel.InitializeAPFEL()
                # collect a_s
                apfel_vals_cur = []
                for Q2 in Q2s:
                    apfel_vals_cur.append(apfel.AlphaQCD(np.sqrt(Q2)) / (4.0 * np.pi))
                # print(apfel_vals_cur)
                np.testing.assert_allclose(apfel_vals, np.array(apfel_vals_cur))
            # check myself to APFEL
            np.testing.assert_allclose(apfel_vals, np.array(my_vals), rtol=2e-4)

    def benchmark_lhapdf_exact(self):
        """test exact towards LHAPDF"""
        Q2s = [1e1, 1e2, 1e3, 1e4]
        alphas_ref = 0.118
        scale_ref = 90 ** 2
        # collect my values
        threshold_holder = thresholds.ThresholdsAtlas.ffns(3)
        # LHAPDF cache
        lhapdf_vals_dict = {
            0: np.array(
                [
                    0.021634715590772086,
                    0.014937700253543466,
                    0.011406683936848657,
                    0.009225845071914008,
                ]
            ),
            1: np.array(
                [
                    0.025027589743439077,
                    0.015730666136188308,
                    0.011590962671090168,
                    0.009215179641099218,
                ]
            ),
            2: np.array(
                [
                    0.02571700975829869,
                    0.015837212322787945,
                    0.011610856755395813,
                    0.009214183512196827,
                ]
            ),
        }
        for order in range(2 + 1):
            sc = StrongCoupling(
                alphas_ref,
                scale_ref,
                threshold_holder,
                order=order,
                method="exact",
            )
            my_vals = []
            for Q2 in Q2s:
                my_vals.append(sc.a_s(Q2))
            lhapdf_vals = lhapdf_vals_dict[order]
            if use_LHAPDF:
                as_lhapdf = lhapdf.mkBareAlphaS("ODE")
                as_lhapdf.setOrderQCD(1 + order)
                as_lhapdf.setFlavorScheme("FIXED", 3)
                as_lhapdf.setAlphaSMZ(alphas_ref)
                as_lhapdf.setMZ(np.sqrt(scale_ref))
                # collect a_s
                lhapdf_vals_cur = []
                for Q2 in Q2s:
                    lhapdf_vals_cur.append(as_lhapdf.alphasQ2(Q2) / (4.0 * np.pi))
                # print(lhapdf_vals_cur)
                np.testing.assert_allclose(lhapdf_vals, np.array(lhapdf_vals_cur))
            # check
            np.testing.assert_allclose(lhapdf_vals, np.array(my_vals), rtol=2e-4)

    def benchmark_lhapdf_zmvfns_lo(self):
        """test ZM-VFNS LO towards LHAPDF"""
        Q2s = [1, 1e1, 1e2, 1e3, 1e4]
        alphas_ref = 0.118
        scale_ref = 900
        m2c = 2
        m2b = 25
        m2t = 1500
        threshold_list = [m2c, m2b, m2t]
        # compute all Lambdas
        # Lambda2_5 = self._get_Lambda2_LO(alphas_ref / (4.0 * np.pi), scale_ref, 5)
        # as_FFNS_LO_5 = StrongCoupling(
        #    alphas_ref, scale_ref, 0, "FFNS", nf=5, method="expanded"
        # )
        # Lambda2_6 = self._get_Lambda2_LO(as_FFNS_LO_5.a_s(m2t), m2t, 6)
        # as_b = as_FFNS_LO_5.a_s(m2b)
        # Lambda2_4 = self._get_Lambda2_LO(as_b, m2b, 4)
        # as_FFNS_LO_4 = StrongCoupling(
        #    as_b * 4.0 * np.pi, m2b, 0, "FFNS", nf=4, method="expanded"
        # )
        # Lambda2_3 = self._get_Lambda2_LO(as_FFNS_LO_4.a_s(m2c), m2c, 3)

        # collect my values
        threshold_holder = thresholds.ThresholdsAtlas(threshold_list)
        as_VFNS_LO = StrongCoupling(alphas_ref, scale_ref, threshold_holder, order=0)
        my_vals = []
        for Q2 in Q2s:
            my_vals.append(as_VFNS_LO.a_s(Q2))
        # LHAPDF cache
        lhapdf_vals = np.array(
            [
                0.01932670387675251,
                0.014008394237618302,
                0.011154570468393434,
                0.009319430765984453,
                0.008084615274633044,
            ]
        )
        if use_LHAPDF:
            # run lhapdf - actually, let's use a different implementation here!
            # as_lhapdf = lhapdf.mkBareAlphaS("analytic")
            as_lhapdf = lhapdf.mkBareAlphaS("ODE")
            as_lhapdf.setOrderQCD(1)
            as_lhapdf.setFlavorScheme("VARIABLE", -1)
            as_lhapdf.setAlphaSMZ(alphas_ref)
            as_lhapdf.setMZ(np.sqrt(scale_ref))
            for k in range(3):
                as_lhapdf.setQuarkMass(1 + k, 0)
            for k, m2 in enumerate(threshold_list):
                as_lhapdf.setQuarkMass(4 + k, np.sqrt(m2))
            # as_lhapdf.setLambda(3, np.sqrt(Lambda2_3))
            # as_lhapdf.setLambda(4, np.sqrt(Lambda2_4))
            # as_lhapdf.setLambda(5, np.sqrt(Lambda2_5))
            # as_lhapdf.setLambda(6, np.sqrt(Lambda2_6))
            # collect a_s
            lhapdf_vals_cur = []
            for Q2 in Q2s:
                lhapdf_vals_cur.append(as_lhapdf.alphasQ2(Q2) / (4.0 * np.pi))
            # print(lhapdf_vals_cur)
            np.testing.assert_allclose(lhapdf_vals, np.array(lhapdf_vals_cur))
        # check - tolerance is determined from
        # Max absolute difference: 2.58611282e-06
        # Max relative difference: 0.00013379
        np.testing.assert_allclose(lhapdf_vals, np.array(my_vals), rtol=1.5e-4)
