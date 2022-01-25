# -*- coding: utf-8 -*-
"""
    Tests for the threshold class
"""
import numpy as np
import pytest

from eko.strong_coupling import StrongCoupling

from eko.msbar_masses import evolve_msbar_mass
from eko.runner import compute_msbar_mass
from eko.evolution_operator.flavors import quark_names

theory_dict = {
    "alphas": 0.1180,
    "Qref": 91,
    "nfref": 5,
    "MaxNfPdf": 6,
    "MaxNfAs": 6,
    "Q0": 1,
    "fact_to_ren_scale_ratio": 1.0,
    "mc": 2.0,
    "mb": 4.0,
    "mt": 175.0,
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
    "HQ": "MSBAR",
    "Qmc": 2.1,
    "Qmb": 4.1,
    "Qmt": 174.9,
    "PTO": 2,
    "ModEv": "TRN",
}


class TestMsbarMasses:
    def test_compute_msbar_mass(self):
        # Test solution of msbar(m) = m
        for method in ["EXA", "EXP"]:
            for order in [1, 2, 3]:
                theory_dict.update({"ModEv": method, "PTO": order})
                strong_coupling = StrongCoupling.from_dict(theory_dict)

                # compute the scale such msbar(m) = m
                m2_computed = compute_msbar_mass(theory_dict)
                m2_test = []
                for nf in [3, 4, 5]:
                    # compute msbar( m )
                    m2_ref = theory_dict[f"m{quark_names[nf]}"] ** 2
                    Q2m_ref = theory_dict[f"Qm{quark_names[nf]}"] ** 2
                    m2_test.append(
                        evolve_msbar_mass(
                            m2_ref,
                            Q2m_ref,
                            dict(
                                fact_to_ren=theory_dict["fact_to_ren_scale_ratio"]
                            ),
                            strong_coupling=strong_coupling,
                            q2_to=m2_computed[nf - 3],
                        )
                    )
                np.testing.assert_allclose(m2_computed, m2_test, rtol=5e-3)

    def test_compute_msbar_mass_VFNS(self):
        # test the solution now with some initial contition
        # not given in the target patch (Qmc, mc are in NF=5)
        theory_dict.update(
            {
                "ModEv": "TRN",
                "PTO": 2,
                "mc": 2.0,
                "mb": 4.0,
                "Qmc": 80.0,
                "Qmb": 85.0,
            }
        )
        strong_coupling = StrongCoupling.from_dict(theory_dict)
        # compute the scale such msbar(m) = m
        m2_computed = compute_msbar_mass(theory_dict)
        m2_test = []
        for nf in [3, 4, 5]:
            # compute msbar( m )
            m2_ref = theory_dict[f"m{quark_names[nf]}"] ** 2
            Q2m_ref = theory_dict[f"Qm{quark_names[nf]}"] ** 2
            m2_test.append(
                evolve_msbar_mass(
                    m2_ref,
                    Q2m_ref,
                    dict(fact_to_ren=theory_dict["fact_to_ren_scale_ratio"]),
                    strong_coupling=strong_coupling,
                    q2_to=m2_computed[nf - 3],
                )
            )
        np.testing.assert_allclose(m2_computed, m2_test, rtol=5e-3)

    def test_errors(self):

        # test mass ordering
        with pytest.raises(ValueError, match="do not preserve the correct ordering"):
            theory_dict.update(
                dict(
                    mc=1.0,
                    mb=1.00001,
                    Qmc=1.0,
                    Qmb=1.00001,
                )
            )
            compute_msbar_mass(theory_dict)
        with pytest.raises(ValueError, match="masses need to be sorted"):
            theory_dict.update(
                dict(
                    mc=1.1,
                    mb=1.0,
                    Qmc=1.2,
                    Qmb=1.0,
                )
            )
            compute_msbar_mass(theory_dict)

        # test forward conditions on alphas_ref
        with pytest.raises(ValueError, match="should be lower than"):
            theory_dict.update(dict(Qmb=91.0001))
            compute_msbar_mass(theory_dict)

        # test backward conditions on alphas_ref
        with pytest.raises(ValueError, match="should be greater than"):
            theory_dict.update(dict(Qmt=89.9999))
            compute_msbar_mass(theory_dict)

        theory_dict.update(dict(Qmb=4.0, Qmt=175))

        # test forward conditions on masses
        with pytest.raises(ValueError, match="should be lower than m"):
            theory_dict.update(dict(mt=174))
            compute_msbar_mass(theory_dict)

        # test backward conditions on masses
        with pytest.raises(ValueError, match="should be greater than m"):
            theory_dict.update(dict(mb=4.1, mt=176))
            compute_msbar_mass(theory_dict)
