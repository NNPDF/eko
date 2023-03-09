"""This module benchmarks MSbar mass evolution against APFEL."""
import numpy as np
import pytest

from eko import msbar_masses
from eko.couplings import Couplings, couplings_mod_ev
from eko.io import types
from eko.io.runcards import OperatorCard, TheoryCard
from eko.quantities.heavy_quarks import QuarkMassRef, QuarkMassScheme

# try to load APFEL - if not available, we'll use the cached values
try:
    import apfel

    use_APFEL = True
except ImportError:
    use_APFEL = False


def update_theory(theory: TheoryCard):
    theory.order = (3, 0)
    theory.couplings.alphas.scale = 91
    theory.couplings.alphaem.value = 0.007496
    theory.couplings.num_flavs_ref = 5
    theory.quark_masses_scheme = QuarkMassScheme.MSBAR
    theory.quark_masses.c = QuarkMassRef([1.5, 18])
    theory.quark_masses.b = QuarkMassRef([4.1, 20])
    theory.quark_masses.t = QuarkMassRef([175.0, 175.0])


@pytest.mark.isolated
class BenchmarkMSbar:
    def benchmark_APFEL_msbar_evolution(
        self, theory_card: TheoryCard, operator_card: OperatorCard
    ):
        update_theory(theory_card)
        bench_values = dict(zip(np.power([1, 96, 150], 2), [3, 5, 5]))
        theory_card.quark_masses.c = QuarkMassRef([1.4, 2.0])
        theory_card.quark_masses.b = QuarkMassRef([4.5, 4.5])
        coupl = theory_card.couplings
        qmasses = theory_card.quark_masses

        apfel_vals_dict = {
            "exact": {
                1: np.array(
                    [
                        [1.6046128320144937, 5.82462147889985, 319.1659462836651],
                        [0.9184216270407894, 3.3338000474743867, 182.67890037277968],
                        [0.8892735505271812, 3.2279947658871553, 176.88119438599423],
                    ]
                ),
                2: np.array(
                    [
                        [1.7606365126844892, 6.707425163030458, 392.9890591164956],
                        [0.8227578662769597, 3.134427109507681, 183.6465604399247],
                        [0.7934449726444709, 3.0227549733595973, 177.10367302092334],
                    ]
                ),
                3: np.array(
                    [
                        [1.8315619347807859, 7.058348563796064, 416.630860855048],
                        [0.8078493428925942, 3.113242546931765, 183.76436225206226],
                        [0.7788276351007536, 3.0014003869088755, 177.16269119698978],
                    ]
                ),
            },
            "expanded": {
                1: np.array(
                    [
                        [1.6046128320144941, 5.824621478899853, 319.1659462836651],
                        [0.9184216270407891, 3.3338000474743863, 182.67890037277968],
                        [0.8892735505271808, 3.227994765887154, 176.88119438599423],
                    ]
                ),
                2: np.array(
                    [
                        [1.7533055503995305, 6.672122790503439, 390.2255025944903],
                        [0.8251533585949282, 3.1400827586994855, 183.65075271254497],
                        [0.7957427000040153, 3.028161864236207, 177.104951823859],
                    ]
                ),
                3: np.array(
                    [
                        [1.8268480938423455, 7.037891257825139, 415.2638988980831],
                        [0.8084237419306857, 3.1144420987483485, 183.76461377981423],
                        [0.7793806824526442, 3.002554084550691, 177.16277079680097],
                    ]
                ),
            },
        }
        # collect my values
        for method in ["exact", "expanded"]:
            operator_card.configs.evolution_method = (
                types.EvolutionMethod.ITERATE_EXPANDED
                if method == "expanded"
                else types.EvolutionMethod.ITERATE_EXACT
            )
            couplevmeth = (
                types.CouplingEvolutionMethod.EXPANDED
                if method == "expanded"
                else types.CouplingEvolutionMethod.EXACT
            )
            for order in [1, 2, 3]:
                theory_card.order = (order, 0)
                as_VFNS = Couplings(
                    couplings=theory_card.couplings,
                    order=theory_card.order,
                    masses=msbar_masses.compute(
                        theory_card.quark_masses,
                        couplings=theory_card.couplings,
                        order=theory_card.order,
                        evmeth=couplevmeth,
                        matching=np.power(list(iter(theory_card.matching)), 2.0),
                        xif2=theory_card.xif**2,
                    ).tolist(),
                    thresholds_ratios=np.power(list(iter(theory_card.matching)), 2.0),
                    method=couplevmeth,
                    hqm_scheme=theory_card.quark_masses_scheme,
                )
                my_vals = []
                for Q2, nf_to in bench_values.items():
                    my_masses = []
                    for n in range(3):
                        my_masses.append(
                            msbar_masses.evolve(
                                qmasses[n].value ** 2,
                                qmasses[n].scale ** 2,
                                strong_coupling=as_VFNS,
                                xif2=1.0,
                                q2_to=Q2,
                                nf_ref=n + 3,
                                nf_to=nf_to,
                            )
                        )
                    my_vals.append(my_masses)
                # get APFEL numbers - if available else use cache
                apfel_vals = apfel_vals_dict[method][order]
                if use_APFEL:
                    # run apfel
                    apfel.CleanUp()
                    apfel.SetTheory("QCD")
                    apfel.SetPerturbativeOrder(order - 1)
                    apfel.SetAlphaEvolution(method)
                    apfel.SetAlphaQCDRef(coupl.alphas.value, coupl.alphas.scale)
                    apfel.SetVFNS()
                    apfel.SetMSbarMasses(
                        qmasses.c.value, qmasses.b.value, qmasses.t.value
                    )
                    apfel.SetMassScaleReference(
                        qmasses.c.scale, qmasses.b.scale, qmasses.t.scale
                    )
                    apfel.SetRenFacRatio(1.0)
                    apfel.InitializeAPFEL()
                    # collect apfel masses
                    apfel_vals_cur = []
                    for Q2 in bench_values:
                        masses = []
                        for n in [4, 5, 6]:
                            masses.append(apfel.HeavyQuarkMass(n, np.sqrt(Q2)))
                        apfel_vals_cur.append(masses)
                    print(apfel_vals_cur)
                    np.testing.assert_allclose(
                        apfel_vals,
                        np.array(apfel_vals_cur),
                        err_msg=f"order={order - 1}",
                    )
                # check myself to APFEL
                np.testing.assert_allclose(
                    apfel_vals, np.sqrt(np.array(my_vals)), rtol=2.3e-3
                )

    def benchmark_APFEL_msbar_solution(
        self, theory_card: TheoryCard, operator_card: OperatorCard
    ):
        update_theory(theory_card)
        apfel_vals_dict = {
            "EXA": {
                1: np.array(
                    [1.9855, 4.8062, 175.0000],
                ),
                2: np.array(
                    [2.1308, 4.9656, 175.0000],
                ),
                3: np.array([2.1566, 4.9841, 175.0000]),
            },
        }
        # collect my values
        theory = theory_card
        operator = operator_card
        coupl = theory_card.couplings
        qmasses = theory_card.quark_masses
        for order in [1, 2, 3]:
            theory.order = (order, 0)
            my_masses = msbar_masses.compute(
                theory.quark_masses,
                couplings=theory_card.couplings,
                order=theory_card.order,
                evmeth=types.CouplingEvolutionMethod.EXACT,
                matching=np.power(list(iter(theory_card.matching)), 2.0),
                xif2=theory_card.xif**2,
            )
            # get APFEL numbers - if available else use cache
            apfel_vals = apfel_vals_dict[
                couplings_mod_ev(operator.configs.evolution_method).value[:3].upper()
            ][order]
            if use_APFEL:
                # run apfel
                apfel.CleanUp()
                apfel.SetTheory("QCD")
                apfel.SetPerturbativeOrder(order - 1)
                apfel.SetAlphaEvolution("exact")
                apfel.SetAlphaQCDRef(coupl.alphas.value, coupl.alphas.scale)
                apfel.SetVFNS()
                apfel.SetMSbarMasses(qmasses.c.value, qmasses.b.value, qmasses.t.value)
                apfel.SetMassScaleReference(
                    qmasses.c.scale, qmasses.b.scale, qmasses.t.scale
                )
                apfel.SetRenFacRatio(theory.xif**2)
                apfel.InitializeAPFEL()
            # check myself to APFEL
            np.testing.assert_allclose(
                apfel_vals, np.sqrt(np.array(my_masses)), rtol=4e-4
            )

    def benchmark_msbar_solution_kthr(self, theory_card: TheoryCard):
        """
        With this test you can see that in EKO
        the solution value of mb is not affected by "kbThr",
        since mb is searched with an Nf=5 larger range.
        While in Apfel this doesn't happen.
        """
        update_theory(theory_card)
        theory_card.order = (1, 0)
        theory_card.matching.c = 1.2
        theory_card.matching.b = 1.8
        theory_card.matching.t = 1.0
        my_masses_thr = msbar_masses.compute(
            theory_card.quark_masses,
            couplings=theory_card.couplings,
            order=theory_card.order,
            evmeth=types.CouplingEvolutionMethod.EXACT,
            matching=np.power(list(iter(theory_card.matching)), 2.0),
            xif2=theory_card.xif**2,
        )
        apfel_masses_thr = [1.9891, 4.5102, 175.0000]
        theory_card.matching.c = 1.0
        theory_card.matching.b = 1.0
        my_masses_plain = msbar_masses.compute(
            theory_card.quark_masses,
            couplings=theory_card.couplings,
            order=theory_card.order,
            evmeth=types.CouplingEvolutionMethod.EXACT,
            matching=np.power(list(iter(theory_card.matching)), 2.0),
            xif2=theory_card.xif**2,
        )

        apfel_masses_plain = ([1.9855, 4.8062, 175.0000],)

        # Eko bottom mass is the same
        np.testing.assert_allclose(my_masses_thr[1], my_masses_plain[1])
        # Eko charm mass is not the same
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_allclose,
            my_masses_thr[0],
            my_masses_plain[0],
        )

        # Apfel bottom masses are not the same
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_allclose,
            apfel_masses_thr[0],
            apfel_masses_plain[0],
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_allclose,
            apfel_masses_thr[0],
            apfel_masses_plain[0],
        )
