"""This module benchmarks alpha_s against LHAPDF and APFEL."""

import numpy as np
import pytest

from eko import matchings
from eko.beta import beta_qcd
from eko.couplings import Couplings
from eko.io.runcards import TheoryCard
from eko.io.types import FlavorsNumber
from eko.quantities.couplings import CouplingEvolutionMethod, CouplingsInfo
from eko.quantities.heavy_quarks import QuarkMassScheme

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

# try to load pegasus - if not available, we'll use the cached values
try:
    import pegasus

    use_PEGASUS = True
except ImportError:
    use_PEGASUS = False


def ref_couplings(ref_values, ref_scale: float, ref_nf: FlavorsNumber) -> CouplingsInfo:
    return CouplingsInfo(
        alphas=ref_values[0],
        alphaem=ref_values[1],
        ref=(ref_scale, ref_nf),
    )


@pytest.mark.isolated
class BenchmarkCouplings:
    def test_a_s(self):
        """Tests the value of alpha_s (for now only at LO) for a given set of
        parameters."""
        # source: JCM
        known_val = 0.0091807954
        ref_mu2 = 90
        ask_q2 = 125
        ref = ref_couplings([0.1181, 0.007496], np.sqrt(ref_mu2), 5)
        as_FFNS_LO = Couplings(
            couplings=ref,
            order=(1, 0),
            method=CouplingEvolutionMethod.EXACT,
            masses=[0, 0, np.inf],
            hqm_scheme=QuarkMassScheme.POLE,
            thresholds_ratios=[1.0, 1.0, 1.0],
        )
        # check local
        np.testing.assert_approx_equal(
            as_FFNS_LO.a(ref_mu2)[0], ref.alphas / 4.0 / np.pi
        )
        # check high
        result = as_FFNS_LO.a(ask_q2)[0]
        np.testing.assert_approx_equal(result, known_val, significant=7)

    def benchmark_LHA_paper(self):
        """Check to :cite:`Giele:2002hx` and :cite:`Dittmar:2005ed`"""
        # LO - FFNS
        # note that the LO-FFNS value reported in :cite:`Giele:2002hx`
        # was corrected in :cite:`Dittmar:2005ed`
        coupling_ref = ref_couplings([0.35, 0.007496], np.sqrt(2), 4)
        as_FFNS_LO = Couplings(
            couplings=coupling_ref,
            order=(1, 0),
            method=CouplingEvolutionMethod.EXACT,
            masses=[0, np.inf, np.inf],
            hqm_scheme=QuarkMassScheme.POLE,
            thresholds_ratios=[1.0, 1.0, 1.0],
        )
        me = as_FFNS_LO.a(1e4)[0] * 4 * np.pi
        ref = 0.117574
        np.testing.assert_approx_equal(me, ref, significant=6)
        # LO - VFNS
        threshold_list = [2, pow(4.5, 2), pow(175, 2)]
        as_VFNS_LO = Couplings(
            couplings=coupling_ref,
            order=(1, 0),
            method=CouplingEvolutionMethod.EXACT,
            masses=threshold_list,
            hqm_scheme=QuarkMassScheme.POLE,
            thresholds_ratios=[1.0, 1.0, 1.0],
        )
        me = as_VFNS_LO.a(1e4)[0] * 4 * np.pi
        ref = 0.122306
        np.testing.assert_approx_equal(me, ref, significant=6)

    def benchmark_APFEL_ffns(self):
        Q2s = [1e1, 1e2, 1e3, 1e4]
        coupling_ref = np.array([0.118, 0.007496])
        scale_ref = 91.0
        nf = 4
        apfel_vals_dict = {
            1: np.array(
                [
                    0.01980124164841284,
                    0.014349241933308077,
                    0.01125134009147229,
                    0.009253560532725833,
                ]
            ),
            2: np.array(
                [
                    0.021603236069744878,
                    0.014887068327193985,
                    0.01139235082295531,
                    0.009245832041857378,
                ]
            ),
            3: np.array(
                [
                    0.02204088975651748,
                    0.014966690529271008,
                    0.011406500825908607,
                    0.009245271638953058,
                ]
            ),
        }
        # collect my values
        threshold_holder = matchings.Atlas.ffns(nf, 0.0)
        for order in [1, 2, 3]:
            as_FFNS = Couplings(
                couplings=ref_couplings(coupling_ref, scale_ref, nf),
                order=(order, 0),
                method=CouplingEvolutionMethod.EXPANDED,
                masses=threshold_holder.walls[1:-1],
                hqm_scheme=QuarkMassScheme.POLE,
                thresholds_ratios=[1.0, 1.0, 1.0],
            )
            my_vals = []
            for Q2 in Q2s:
                my_vals.append(as_FFNS.a(Q2)[0])
            # get APFEL numbers - if available else use cache
            apfel_vals = apfel_vals_dict[order]
            if use_APFEL:
                # run apfel
                apfel.CleanUp()
                apfel.SetTheory("QCD")
                apfel.SetPerturbativeOrder(order - 1)
                apfel.SetAlphaEvolution("expanded")
                apfel.SetAlphaQCDRef(coupling_ref[0], scale_ref)
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

    def benchmark_pegasus_ffns(self):
        Q2s = [1e1, 1e2, 1e3, 1e4]
        coupling_ref = np.array([0.118, 0.007496])
        scale_ref = 91.0
        nf = 4
        pegasus_vals_dict = {
            1: np.array(
                [
                    0.01980124164841284,
                    0.014349241933308077,
                    0.011251340091472288,
                    0.009253560532725833,
                ]
            ),
            2: np.array(
                [
                    0.021869297350539534,
                    0.014917847337312901,
                    0.011395004941942033,
                    0.00924584172632009,
                ]
            ),
            3: np.array(
                [
                    0.022150834782048476,
                    0.014974959766044743,
                    0.011407029399653526,
                    0.009245273177012448,
                ]
            ),
            4: np.array(
                [
                    0.02224042559454458,
                    0.014988806602725375,
                    0.01140950716164355,
                    0.009245168439298405,
                ]
            ),
        }
        # collect my values
        threshold_holder = matchings.Atlas.ffns(nf, 0.0)
        couplings = ref_couplings(coupling_ref, scale_ref, nf)
        for order in [1, 2, 3, 4]:
            as_FFNS = Couplings(
                couplings=couplings,
                order=(order, 0),
                method=CouplingEvolutionMethod.EXACT,
                masses=threshold_holder.walls[1:-1],
                hqm_scheme=QuarkMassScheme.POLE,
                thresholds_ratios=[1.0, 1.0, 1.0],
            )
            my_vals = []
            for Q2 in Q2s:
                my_vals.append(as_FFNS.a(Q2)[0])
            # get APFEL numbers - if available else use cache
            pegasus_vals = pegasus_vals_dict[order]
            if use_PEGASUS:
                # run pegasus
                pegasus.colour.ca = 3.0
                pegasus.colour.cf = 4.0 / 3.0
                pegasus.colour.tr = 0.5
                pegasus.betafct()
                pegasus.aspar.naord = order - 1
                pegasus.aspar.nastps = 20
                # collect a_s
                pegasus_vals_cur = []
                for Q2 in Q2s:
                    pegasus_vals_cur.append(
                        pegasus.__getattribute__("as")(
                            Q2, scale_ref**2.0, coupling_ref[0] / (4.0 * np.pi), nf
                        )
                    )
                # print(pegasus_vals_cur)
                np.testing.assert_allclose(
                    pegasus_vals,
                    np.array(pegasus_vals_cur),
                    err_msg=f"order={order - 1}",
                )
            # check myself to PEGASUS
            np.testing.assert_allclose(
                pegasus_vals, np.array(my_vals), rtol=1.5e-7, err_msg=f"order={order}"
            )

    def benchmark_APFEL_vfns(self):
        Q2s = [1, 2**2, 3**2, 90**2, 100**2]
        coupling_ref = np.array([0.118, 0.007496])
        scale_ref = 91.0
        nf_ref = 5
        threshold_list = np.power([2, 4, 175], 2)
        apfel_vals_dict = {
            1: np.array(
                [
                    0.028938898786215545,
                    0.021262022520127353,
                    0.018590827846469413,
                    0.009405104970805002,
                    0.00926434063784546,
                ]
            ),
            2: np.array(
                [
                    0.035670881093047654,
                    0.02337584106433519,
                    0.01985110421500437,
                    0.009405815313164215,
                    0.009258502199861199,
                ]
            ),
            3: np.array(
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
        for order in [1, 2, 3]:
            as_VFNS = Couplings(
                couplings=ref_couplings(coupling_ref, scale_ref, nf_ref),
                order=(order, 0),
                method=CouplingEvolutionMethod.EXPANDED,
                masses=threshold_list,
                hqm_scheme=QuarkMassScheme.POLE,
                thresholds_ratios=[1.0, 1.0, 1.0],
            )
            my_vals = []
            for Q2 in Q2s:
                my_vals.append(as_VFNS.a(Q2)[0])
            # get APFEL numbers - if available else use cache
            apfel_vals = apfel_vals_dict[order]
            if use_APFEL:
                # run apfel
                apfel.CleanUp()
                apfel.SetTheory("QCD")
                apfel.SetPerturbativeOrder(order - 1)
                apfel.SetAlphaEvolution("expanded")
                apfel.SetAlphaQCDRef(coupling_ref[0], scale_ref)
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

    def benchmark_APFEL_vfns_fact_to_ren(self):
        Q2s = [
            1.5**2,
            2**2,
            3**2,
            4**2,
            70**2,
            80**2,
            90**2,
            100**2,
            110**2,
            120**2,
        ]
        coupling_ref = np.array([0.118, 0.007496])
        scale_ref = 91.0
        nf_refs = (5, 6)
        fact_to_ren_lin_list = [0.567, 2.34]
        masses_list = np.power([2, 2 * 4, 2 * 92], 2)
        apfel_vals_dict_list = [
            {
                1: np.array(
                    [
                        0.02536996716194434,
                        0.02242405760517867,
                        0.019270297766607315,
                        0.017573474616674672,
                        0.009758787415866933,
                        0.00956761747052419,
                        0.009405104970805002,
                        0.00926434063784546,
                        0.009140585174416013,
                        0.009030457524243522,
                    ]
                ),
                2: np.array(
                    [
                        0.02892091611225328,
                        0.024548883236784783,
                        0.020329067356870584,
                        0.018489330048933994,
                        0.009777335619585536,
                        0.009576273535323533,
                        0.009405815393635488,
                        0.009258507721473314,
                        0.009129255945850333,
                        0.009014436496515657,
                    ]
                ),
                3: np.array(
                    [
                        0.029220134213545565,
                        0.024598646506863896,
                        0.020256507668384785,
                        0.018516529262400945,
                        0.00977816967988805,
                        0.009576658129637192,
                        0.009405846634492874,
                        0.009258253518802814,
                        0.009128766134980268,
                        0.009013748776298706,
                    ]
                ),
            },
            {
                1: np.array(
                    [
                        0.023451267902337505,
                        0.021080893784642736,
                        0.018452203115256843,
                        0.0170127532599337,
                        0.00974027424891201,
                        0.009551918909176069,
                        0.009403801918958557,
                        0.009275145889222168,
                        0.009161757943458262,
                        0.009060636882751797,
                    ]
                ),
                2: np.array(
                    [
                        0.028479841693964322,
                        0.02457741561064943,
                        0.020668723336288768,
                        0.01832324913740207,
                        0.00986337072082633,
                        0.009557712951433234,
                        0.009404279433906803,
                        0.009271211089740456,
                        0.009154091133886749,
                        0.009049764354779253,
                    ]
                ),
                3: np.array(
                    [
                        0.029458461672676982,
                        0.025177951225443865,
                        0.021006297788672076,
                        0.018449012475696365,
                        0.009880255980699394,
                        0.009557644276584587,
                        0.009404273824351815,
                        0.009271256953218584,
                        0.009154179878124983,
                        0.00904988942200004,
                    ]
                ),
            },
        ]
        for fact_to_ren_lin, apfel_vals_dict, nf_ref in zip(
            fact_to_ren_lin_list, apfel_vals_dict_list, nf_refs
        ):
            # collect my values
            for order in [1, 2, 3]:
                as_VFNS = Couplings(
                    couplings=ref_couplings(coupling_ref, scale_ref, nf_ref),
                    order=(order, 0),
                    method=CouplingEvolutionMethod.EXACT,
                    masses=masses_list.tolist(),
                    hqm_scheme=QuarkMassScheme.POLE,
                    thresholds_ratios=np.array([1.0, 1.0, 1.0]) / fact_to_ren_lin**2,
                )
                my_vals = []
                for Q2 in Q2s:
                    my_vals.append(as_VFNS.a(Q2)[0])
                # get APFEL numbers - if available else use cache
                apfel_vals = apfel_vals_dict[order]
                if use_APFEL:
                    # run apfel
                    apfel.CleanUp()
                    apfel.SetTheory("QCD")
                    apfel.SetPerturbativeOrder(order - 1)
                    apfel.SetAlphaEvolution("exact")
                    apfel.SetAlphaQCDRef(coupling_ref[0], scale_ref)
                    apfel.SetVFNS()
                    apfel.SetPoleMasses(*np.sqrt(masses_list))
                    apfel.SetRenFacRatio(1.0 / fact_to_ren_lin)
                    apfel.InitializeAPFEL()
                    # collect a_s
                    apfel_vals_cur = []
                    for Q2 in Q2s:
                        apfel_vals_cur.append(
                            apfel.AlphaQCD(np.sqrt(Q2)) / (4.0 * np.pi)
                        )
                    np.testing.assert_allclose(apfel_vals, np.array(apfel_vals_cur))
                # check myself to APFEL
                np.testing.assert_allclose(
                    apfel_vals,
                    np.array(my_vals),
                    rtol=2.5e-5,
                    err_msg=f"{order=},{fact_to_ren_lin=}",
                )

    def benchmark_APFEL_vfns_threshold(self):
        Q2s = np.power([30, 96, 150], 2)
        coupling_ref = np.array([0.118, 0.007496])
        scale_ref = 91.0
        nf_ref = 4
        threshold_list = np.power([30, 95, 240], 2)
        thresholds_ratios = [2.34**2, 1.0**2, 0.5**2]
        apfel_vals_dict = {
            2: np.array(
                [0.011543349125207046, 0.00930916017456183, 0.008683622955304702]
            ),
            3: np.array(
                [0.011530819844835012, 0.009312638352288492, 0.00867793622077099]
            ),
        }
        # collect my values
        for order in [2, 3]:
            as_VFNS = Couplings(
                couplings=ref_couplings(coupling_ref, scale_ref, nf_ref),
                order=(order, 0),
                method=CouplingEvolutionMethod.EXPANDED,
                masses=threshold_list.tolist(),
                hqm_scheme=QuarkMassScheme.POLE,
                thresholds_ratios=thresholds_ratios,
            )
            my_vals = []
            for Q2 in Q2s:
                my_vals.append(as_VFNS.a(Q2)[0])
            # get APFEL numbers - if available else use cache
            apfel_vals = apfel_vals_dict[order]
            if use_APFEL:
                # run apfel
                apfel.CleanUp()
                apfel.SetTheory("QCD")
                apfel.SetPerturbativeOrder(order - 1)
                apfel.SetAlphaEvolution("expanded")
                apfel.SetAlphaQCDRef(coupling_ref[0], scale_ref)
                apfel.SetVFNS()
                apfel.SetPoleMasses(*np.sqrt(threshold_list))
                apfel.SetMassMatchingScales(*np.sqrt(list(thresholds_ratios)))
                apfel.SetRenFacRatio(1)
                apfel.InitializeAPFEL()
                # collect a_s
                apfel_vals_cur = []
                for Q2 in Q2s:
                    apfel_vals_cur.append(apfel.AlphaQCD(np.sqrt(Q2)) / (4.0 * np.pi))
                np.testing.assert_allclose(apfel_vals, np.array(apfel_vals_cur))
            # check myself to APFEL
            np.testing.assert_allclose(apfel_vals, np.array(my_vals))

    def benchmark_APFEL_vfns_msbar(self):
        Q2s = np.power([3, 96, 150], 2)
        coupling_ref = np.array([0.118, 0.007496])
        scale_ref = 91.0
        nf_ref = 5
        Q2m = np.power([2.0, 4.0, 175], 2)
        m2s = np.power((1.4, 4.0, 175), 2)
        apfel_vals_dict = {
            2: np.array(
                [0.01985110421500437, 0.009315017128518715, 0.00873290495922747]
            ),
            3: np.array(
                [0.02005213805142418, 0.009314871933701218, 0.008731890252153528]
            ),
        }
        # collect my values
        for order in [2, 3]:
            as_VFNS = Couplings(
                couplings=ref_couplings(coupling_ref, scale_ref, nf_ref),
                order=(order, 0),
                method=CouplingEvolutionMethod.EXPANDED,
                masses=m2s.tolist(),
                thresholds_ratios=[1.0, 1.0, 1.0],
                hqm_scheme=QuarkMassScheme.MSBAR,
            )
            my_vals = []
            for Q2 in Q2s:
                print(Q2)
                my_vals.append(as_VFNS.a(Q2)[0])
            # get APFEL numbers - if available else use cache
            apfel_vals = apfel_vals_dict[order]
            if use_APFEL:
                # run apfel
                apfel.CleanUp()
                apfel.SetTheory("QCD")
                apfel.SetPerturbativeOrder(order - 1)
                apfel.SetAlphaEvolution("expanded")
                apfel.SetAlphaQCDRef(coupling_ref[0], scale_ref)
                apfel.SetVFNS()
                apfel.SetMSbarMasses(*np.sqrt(m2s))
                apfel.SetMassScaleReference(*np.sqrt(Q2m))
                apfel.SetRenFacRatio(1)
                apfel.InitializeAPFEL()
                # collect a_s
                apfel_vals_cur = []
                for Q2 in Q2s:
                    apfel_vals_cur.append(apfel.AlphaQCD(np.sqrt(Q2)) / (4.0 * np.pi))
                print(apfel_vals_cur)
                np.testing.assert_allclose(apfel_vals, np.array(apfel_vals_cur))
            # check myself to APFEL
            np.testing.assert_allclose(apfel_vals, np.array(my_vals))

    def _get_Lambda2_LO(self, as_ref, scale_ref, nf):
        """Transformation to Lambda_QCD."""
        beta0 = beta_qcd((2, 0), nf)
        return scale_ref * np.exp(-1.0 / (as_ref * beta0))

    def benchmark_lhapdf_ffns_lo(self):
        """Test FFNS LO towards LHAPDF."""
        Q2s = [1, 1e1, 1e2, 1e3, 1e4]
        coupling_ref = np.array([0.118, 0.007496])
        scale_ref = 91.0
        nf = 4
        # collect my values
        threshold_holder = matchings.Atlas.ffns(nf, 0.0)
        as_FFNS_LO = Couplings(
            couplings=ref_couplings(coupling_ref, scale_ref, nf),
            order=(1, 0),
            method=CouplingEvolutionMethod.EXACT,
            masses=threshold_holder.walls[1:-1],
            hqm_scheme=QuarkMassScheme.POLE,
            thresholds_ratios=[1.0, 1.0, 1.0],
        )
        my_vals = []
        for Q2 in Q2s:
            my_vals.append(as_FFNS_LO.a(Q2)[0])
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
            Lambda2 = self._get_Lambda2_LO(
                coupling_ref[0] / (4.0 * np.pi), scale_ref**2, nf
            )
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
        """Test exact towards APFEL."""
        Q2s = [1e1, 1e2, 1e3, 1e4]
        coupling_ref = np.array([0.118, 0.007496])
        scale_ref = 90
        nf = 3
        # collect my values
        threshold_holder = matchings.Atlas.ffns(nf, 0.0)
        # LHAPDF cache
        apfel_vals_dict = {
            1: np.array(
                [
                    0.021635019899707245,
                    0.014937719308417242,
                    0.01140668497649318,
                    0.009225844999427163,
                ]
            ),
            2: np.array(
                [
                    0.025027723843261098,
                    0.015730685887616093,
                    0.01159096381106341,
                    0.009215179564010682,
                ]
            ),
            3: np.array(
                [
                    0.025717091835015565,
                    0.01583723253352162,
                    0.011610857909393214,
                    0.009214183434685514,
                ]
            ),
        }
        for order in range(1, 3 + 1):
            sc = Couplings(
                couplings=ref_couplings(coupling_ref, scale_ref, nf),
                order=(order, 0),
                method=CouplingEvolutionMethod.EXACT,
                masses=threshold_holder.walls[1:-1],
                hqm_scheme=QuarkMassScheme.POLE,
                thresholds_ratios=[1.0, 1.0, 1.0],
            )
            my_vals = []
            for Q2 in Q2s:
                my_vals.append(sc.a(Q2)[0])
            # get APFEL numbers - if available else use cache
            apfel_vals = apfel_vals_dict[order]
            if use_APFEL:
                # run apfel
                apfel.CleanUp()
                apfel.SetTheory("QCD")
                apfel.SetPerturbativeOrder(order - 1)
                apfel.SetAlphaEvolution("exact")
                apfel.SetAlphaQCDRef(coupling_ref[0], scale_ref)
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
        """Test exact towards LHAPDF."""
        Q2s = [1e1, 1e2, 1e3, 1e4]
        coupling_ref = np.array([0.118, 0.007496])
        scale_ref = 90
        nf = 3
        # collect my values
        threshold_holder = matchings.Atlas.ffns(nf, 0.0)
        # LHAPDF cache
        lhapdf_vals_dict = {
            1: np.array(
                [
                    0.021634715590772086,
                    0.014937700253543466,
                    0.011406683936848657,
                    0.009225845071914008,
                ]
            ),
            2: np.array(
                [
                    0.025027589743439077,
                    0.015730666136188308,
                    0.011590962671090168,
                    0.009215179641099218,
                ]
            ),
            3: np.array(
                [
                    0.02571700975829869,
                    0.015837212322787945,
                    0.011610856755395813,
                    0.009214183512196827,
                ]
            ),
            4: np.array(
                [
                    0.025955428065113105,
                    0.015862752802768762,
                    0.011614793662449371,
                    0.009214009540864635,
                ]
            ),
        }
        for order in range(1, 4 + 1):
            sc = Couplings(
                couplings=ref_couplings(coupling_ref, scale_ref, nf),
                order=(order, 0),
                method=CouplingEvolutionMethod.EXACT,
                masses=threshold_holder.walls[1:-1],
                hqm_scheme=QuarkMassScheme.POLE,
                thresholds_ratios=[1.0, 1.0, 1.0],
            )
            my_vals = []
            for Q2 in Q2s:
                my_vals.append(sc.a(Q2)[0])
            lhapdf_vals = lhapdf_vals_dict[order]
            if use_LHAPDF:
                as_lhapdf = lhapdf.mkBareAlphaS("ODE")
                as_lhapdf.setOrderQCD(order)
                as_lhapdf.setFlavorScheme("FIXED", 3)
                as_lhapdf.setAlphaSMZ(coupling_ref[0])
                as_lhapdf.setMZ(scale_ref)
                # collect a_s
                lhapdf_vals_cur = []
                for Q2 in Q2s:
                    lhapdf_vals_cur.append(as_lhapdf.alphasQ2(Q2) / (4.0 * np.pi))
                # print(lhapdf_vals_cur)
                np.testing.assert_allclose(lhapdf_vals, np.array(lhapdf_vals_cur))
            # check
            np.testing.assert_allclose(lhapdf_vals, np.array(my_vals), rtol=2e-4)

    def benchmark_lhapdf_zmvfns_lo(self):
        """Test ZM-VFNS LO towards LHAPDF."""
        Q2s = [1, 1e1, 1e2, 1e3, 1e4]
        coupling_ref = np.array([0.118, 0.007496])
        scale_ref = 900
        nf_ref = 5
        m2c = 2.0
        m2b = 25.0
        m2t = 1500.0
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
        as_VFNS_LO = Couplings(
            couplings=ref_couplings(coupling_ref, np.sqrt(scale_ref), nf_ref),
            order=(1, 0),
            method=CouplingEvolutionMethod.EXACT,
            masses=threshold_list,
            hqm_scheme=QuarkMassScheme.POLE,
            thresholds_ratios=[1.0, 1.0, 1.0],
        )
        my_vals = []
        for Q2 in Q2s:
            my_vals.append(as_VFNS_LO.a(Q2)[0])
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
            as_lhapdf.setAlphaSMZ(coupling_ref[0])
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

    def benchmark_APFEL_fact_to_ren_lha_settings(self, theory_card: TheoryCard):
        theory = theory_card
        theory.order = (3, 0)
        theory.couplings.alphas = 0.35
        theory.couplings.alphaem = 0.007496
        theory.couplings.ref = (float(np.sqrt(2)), 4)
        theory.xif = np.sqrt(1.0 / 2.0)
        theory.heavy.masses.c.value = np.sqrt(2.0)
        theory.heavy.masses.b.value = 4.5
        theory.heavy.masses.t.value = 175.0
        qmasses = theory.heavy.masses

        masses = tuple(mq.value**2 for mq in theory.heavy.masses)

        mu2s = [2.0]
        xif2 = theory.xif**2
        sc = Couplings(
            couplings=theory.couplings,
            order=theory.order,
            method=CouplingEvolutionMethod.EXACT,
            masses=masses,
            hqm_scheme=QuarkMassScheme.POLE,
            thresholds_ratios=np.power(list(iter(theory.heavy.matching_ratios)), 2.0)
            * xif2,
        )
        for mu2 in mu2s:
            my_val = sc.a(mu2 * xif2)[0]
            my_val_4 = sc.a(mu2 * xif2, nf_to=4)[0]
            path_4 = sc.atlas.path((mu2 * xif2, 4))
            my_val_3 = sc.a(mu2 * xif2, nf_to=3)[0]
            path_3 = sc.atlas.path((mu2 * xif2, 3))

            # path_4 it's not matched
            assert len(path_4) == 1

            # path_3 is the same as path backward in nf and in q2.
            assert len(path_3) == 2
            assert path_3[1].nf < path_3[0].nf
            assert path_3[1].origin < path_3[0].origin

            apfel_val_ref = 0.03478112968976964
            if use_APFEL:
                # run apfel
                apfel.CleanUp()
                apfel.SetTheory("QCD")
                apfel.SetPerturbativeOrder(theory.order[0] - 1)
                apfel.SetAlphaEvolution("exact")
                apfel.SetAlphaQCDRef(theory.couplings.alphas, theory.couplings.ref[0])
                apfel.SetVFNS()
                apfel.SetPoleMasses(qmasses.c.value, qmasses.b.value, qmasses.t.value)
                apfel.SetMassMatchingScales(
                    theory.heavy.matching_ratios.c,
                    theory.heavy.matching_ratios.b,
                    theory.heavy.matching_ratios.t,
                )
                apfel.SetRenFacRatio(theory.xif)
                apfel.InitializeAPFEL()
                # collect a_s
                apfel_val = apfel.AlphaQCD(np.sqrt(mu2) * theory.xif) / (4.0 * np.pi)
                # check APFEL cached value
                np.testing.assert_allclose(apfel_val_ref, apfel_val)

            # check myself to APFEL
            np.testing.assert_allclose(apfel_val_ref, my_val, rtol=0.03)
            np.testing.assert_allclose(apfel_val_ref, my_val_4)
            np.testing.assert_allclose(my_val, my_val_3)
