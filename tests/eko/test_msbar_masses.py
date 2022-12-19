"""
    Tests for the threshold class
"""
from math import nan

import numpy as np
import pytest

from eko import msbar_masses
from eko.couplings import Couplings
from eko.io import types
from eko.io.runcards import TheoryCard


@pytest.fixture()
def theory_card(theory_card: TheoryCard):
    th = theory_card
    th.order = (3, 0)
    th.couplings.alphas.value = 0.1180
    th.couplings.alphas.scale = 91.0
    th.couplings.alphaem.value = 0.00781
    th.couplings.num_flavs_ref = 5
    for qname, qmass in zip("cbt", [(2.0, 2.1), (4.0, 4.1), (175.0, 174.9)]):
        q = getattr(th.quark_masses, qname)
        q.value, q.scale = qmass
    th.quark_masses_scheme = types.QuarkMassSchemes.MSBAR

    return th


class TestMsbarMasses:
    @pytest.mark.skip
    def test_compute_msbar_mass(self, theory_card: TheoryCard):
        th = theory_card

        # Test solution of msbar(m) = m
        for method in list(types.CouplingEvolutionMethod):
            for order in [2, 3, 4]:
                theory_card.order = (order, 0)

                # compute the scale such msbar(m) = m
                m2_computed = msbar_masses.compute(
                    th.quark_masses, th.couplings, th.order, method, th.matching
                )
                massesobj = types.HeavyQuarkMasses.from_dict(
                    {q: [np.sqrt(m2q), nan] for m2q, q in zip(m2_computed, "cbt")}
                )
                strong_coupling = Couplings(th.couplings, th.order, method, massesobj)
                m2_test = []
                for nf in [3, 4, 5]:
                    hq = "cbt"[nf - 3]
                    mass = getattr(th.quark_masses, hq)
                    # compute msbar( m )
                    m2_ref = mass.value**2
                    Q2m_ref = mass.scale**2
                    m2_test.append(
                        msbar_masses.evolve(
                            m2_ref,
                            Q2m_ref,
                            strong_coupling=strong_coupling,
                            xif2=th.xif**2,
                            q2_to=m2_computed[nf - 3],
                        )
                    )
                np.testing.assert_allclose(m2_computed, m2_test, rtol=6e-4)

    def test_compute_msbar_mass_VFNS(self, theory_card: TheoryCard):
        # test the solution now with some initial contition
        # not given in the target patch (Qmc, mc are in NF=5)
        th = theory_card
        th.order = (4, 0)
        th.quark_masses.c.value = 2.0
        th.quark_masses.c.scale = 80.0
        th.quark_masses.b.value = 4.0
        th.quark_masses.b.scale = 85.0
        # compute the scale such msbar(m) = m
        m2_computed = msbar_masses.compute(
            th.quark_masses,
            th.couplings,
            th.order,
            types.CouplingEvolutionMethod.EXPANDED,
            th.matching,
        )
        massesobj = types.HeavyQuarkMasses.from_dict(
            {q: [np.sqrt(m2q), nan] for m2q, q in zip(m2_computed, "cbt")}
        )
        strong_coupling = Couplings(
            th.couplings, th.order, types.CouplingEvolutionMethod.EXPANDED, massesobj
        )
        m2_test = []
        for nf in [3, 4, 5]:
            hq = "cbt"[nf - 3]
            mass = getattr(th.quark_masses, hq)
            # compute msbar( m )
            m2_ref = mass.value**2
            Q2m_ref = mass.scale**2
            m2_test.append(
                msbar_masses.evolve(
                    m2_ref,
                    Q2m_ref,
                    strong_coupling=strong_coupling,
                    xif2=th.xif**2,
                    q2_to=m2_computed[nf - 3],
                )
            )
        np.testing.assert_allclose(m2_computed, m2_test, rtol=6e-4)

    def test_errors(self, theory_card: TheoryCard):
        th = theory_card

        def compute(theory: TheoryCard):
            msbar_masses.compute(
                theory.quark_masses,
                theory.couplings,
                theory.order,
                types.CouplingEvolutionMethod.EXPANDED,
                theory.matching,
            )

        # test mass ordering
        with pytest.raises(ValueError, match="Msbar masses are not to be sorted"):
            th.quark_masses.c.value = 1.1
            th.quark_masses.c.scale = 1.2
            th.quark_masses.b.value = 1.0
            th.quark_masses.b.scale = 1.0
            compute(th)

        # test forward conditions on alphas_ref
        with pytest.raises(ValueError, match="should be lower than"):
            th.quark_masses.b.scale = 91.0001
            compute(th)

        # test backward conditions on alphas_ref
        with pytest.raises(ValueError, match="should be greater than"):
            th.quark_masses.b.scale = 89.9999
            compute(th)

        th.quark_masses.b.scale = 4.0
        th.quark_masses.t.scale = 175.0

        # test forward conditions on masses
        with pytest.raises(ValueError, match="should be lower than m"):
            th.quark_masses.t.value = 174.0
            compute(th)

        # test backward conditions on masses
        with pytest.raises(ValueError, match="should be greater than m"):
            th.quark_masses.b.value = 4.1
            th.quark_masses.t.value = 176.0
            compute(th)
