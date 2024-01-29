"""Tests for the threshold class."""

import numpy as np
import pytest

from eko import msbar_masses
from eko.couplings import Couplings
from eko.io import types
from eko.io.runcards import TheoryCard
from eko.quantities.couplings import CouplingEvolutionMethod
from eko.quantities.heavy_quarks import HeavyQuarkMasses, QuarkMassRef, QuarkMassScheme


@pytest.fixture()
def theory_card(theory_card: TheoryCard):
    th = theory_card
    th.order = (3, 0)
    th.couplings.alphas = 0.1180
    th.couplings.scale = 91.0
    th.couplings.alphaem = 0.00781
    th.couplings.num_flavs_ref = 5
    th.heavy.masses = HeavyQuarkMasses(
        [QuarkMassRef(val) for val in [(2.0, 2.1), (4.0, 4.1), (175.0, 174.9)]]
    )

    th.heavy.masses_scheme = QuarkMassScheme.MSBAR

    return th


class TestMSbarMasses:
    def test_compute_msbar_mass(self, theory_card: TheoryCard):
        th = theory_card

        # Test solution of msbar(m) = m
        for method in list(CouplingEvolutionMethod):
            for order in [2, 3, 4]:
                theory_card.order = (order, 0)

                # compute the scale such msbar(m) = m
                m2_computed = msbar_masses.compute(
                    th.heavy.masses,
                    th.couplings,
                    th.order,
                    method,
                    np.power(th.heavy.matching_ratios, 2.0),
                )
                strong_coupling = Couplings(
                    th.couplings,
                    th.order,
                    method,
                    masses=m2_computed.tolist(),
                    hqm_scheme=QuarkMassScheme.POLE,
                    thresholds_ratios=[1.0, 1.0, 1.0],
                )
                m2_test = []
                for nf in [3, 4, 5]:
                    hq = "cbt"[nf - 3]
                    mass = getattr(th.heavy.masses, hq)
                    # compute msbar( m )
                    m2_ref = mass.value**2
                    Q2m_ref = mass.scale**2
                    m2_test.append(
                        msbar_masses.evolve(
                            m2_ref,
                            Q2m_ref,
                            strong_coupling=strong_coupling,
                            thresholds_ratios=th.heavy.matching_ratios,
                            xif2=th.xif**2,
                            q2_to=m2_computed[nf - 3],
                        )
                    )
                np.testing.assert_allclose(
                    m2_computed, m2_test, rtol=6e-4, err_msg=f"{method=},{order=}"
                )

    def test_compute_msbar_mass_VFNS(self, theory_card: TheoryCard):
        # test the solution now with some initial contition
        # not given in the target patch (Qmc, mc are in NF=5)
        th = theory_card
        th.order = (4, 0)
        th.heavy.masses.c.value = 2.0
        th.heavy.masses.c.scale = 80.0
        th.heavy.masses.b.value = 4.0
        th.heavy.masses.b.scale = 85.0
        # compute the scale such msbar(m) = m
        m2_computed = msbar_masses.compute(
            th.heavy.masses,
            th.couplings,
            th.order,
            CouplingEvolutionMethod.EXPANDED,
            np.power(th.heavy.matching_ratios, 2.0),
        )
        strong_coupling = Couplings(
            th.couplings,
            th.order,
            CouplingEvolutionMethod.EXPANDED,
            m2_computed.tolist(),
            hqm_scheme=QuarkMassScheme.MSBAR,
            thresholds_ratios=[1.0, 1.0, 1.0],
        )
        m2_test = []
        for nf in [3, 4, 5]:
            hq = "cbt"[nf - 3]
            mass = getattr(th.heavy.masses, hq)
            # compute msbar( m )
            m2_ref = mass.value**2
            Q2m_ref = mass.scale**2
            m2_test.append(
                msbar_masses.evolve(
                    m2_ref,
                    Q2m_ref,
                    strong_coupling=strong_coupling,
                    thresholds_ratios=th.heavy.matching_ratios,
                    xif2=th.xif**2,
                    q2_to=m2_computed[nf - 3],
                )
            )
        np.testing.assert_allclose(m2_computed, m2_test, rtol=6e-4)

    def test_errors(self, theory_card: TheoryCard):
        th = theory_card

        def compute(theory: TheoryCard):
            msbar_masses.compute(
                theory.heavy.masses,
                theory.couplings,
                theory.order,
                CouplingEvolutionMethod.EXPANDED,
                np.power(theory.heavy.matching_ratios, 2.0),
            )

        # test mass ordering
        with pytest.raises(ValueError, match="MSbar masses are not to be sorted"):
            th.heavy.masses.c.value = 1.1
            th.heavy.masses.c.scale = 1.2
            th.heavy.masses.b.value = 1.0
            th.heavy.masses.b.scale = 1.0
            compute(th)

        # test forward conditions on alphas_ref
        with pytest.raises(ValueError, match="should be lower than"):
            th.heavy.masses.b.scale = 91.0001
            compute(th)

        # test backward conditions on alphas_ref
        with pytest.raises(ValueError, match="should be greater than"):
            th.heavy.masses.t.scale = 89.9999
            compute(th)

        th.heavy.masses.b.scale = 4.0
        th.heavy.masses.t.scale = 175.0

        # test forward conditions on masses
        with pytest.raises(ValueError, match="should be lower than m"):
            th.heavy.masses.t.value = 174.0
            compute(th)

        # test backward conditions on masses
        with pytest.raises(ValueError, match="should be greater than m"):
            th.heavy.masses.b.value = 4.1
            th.heavy.masses.t.value = 176.0
            compute(th)
