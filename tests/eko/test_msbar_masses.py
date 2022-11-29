"""
    Tests for the threshold class
"""
import numpy as np
import pytest

from eko import msbar_masses
from eko.basis_rotation import quark_names
from eko.couplings import Couplings

theory_dict = {
    "alphas": 0.1180,
    "alphaem": 0.00781,
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
    "order": (3, 0),
    "ModEv": "TRN",
    "ModSV": None,
}


class TestMsbarMasses:
    def test_compute_msbar_mass(self):
        # Test solution of msbar(m) = m
        for method in ["EXA", "EXP"]:
            for order in [2, 3, 4]:
                theory_dict.update({"ModEv": method, "order": (order, 0)})

                # compute the scale such msbar(m) = m
                m2_computed = msbar_masses.compute(theory_dict)
                strong_coupling = Couplings.from_dict(theory_dict, m2_computed)
                m2_test = []
                for nf in [3, 4, 5]:
                    # compute msbar( m )
                    m2_ref = theory_dict[f"m{quark_names[nf]}"] ** 2
                    Q2m_ref = theory_dict[f"Qm{quark_names[nf]}"] ** 2
                    m2_test.append(
                        msbar_masses.evolve(
                            m2_ref,
                            Q2m_ref,
                            strong_coupling=strong_coupling,
                            fact_to_ren=theory_dict["fact_to_ren_scale_ratio"] ** 2,
                            q2_to=m2_computed[nf - 3],
                        )
                    )
                np.testing.assert_allclose(m2_computed, m2_test, rtol=6e-4)

    def test_compute_msbar_mass_VFNS(self):
        # test the solution now with some initial contition
        # not given in the target patch (Qmc, mc are in NF=5)
        theory_dict.update(
            {
                "ModEv": "TRN",
                "order": (4, 0),
                "mc": 2.0,
                "mb": 4.0,
                "Qmc": 80.0,
                "Qmb": 85.0,
            }
        )
        # compute the scale such msbar(m) = m
        m2_computed = msbar_masses.compute(theory_dict)
        strong_coupling = Couplings.from_dict(theory_dict, m2_computed)
        m2_test = []
        for nf in [3, 4, 5]:
            # compute msbar( m )
            m2_ref = theory_dict[f"m{quark_names[nf]}"] ** 2
            Q2m_ref = theory_dict[f"Qm{quark_names[nf]}"] ** 2
            m2_test.append(
                msbar_masses.evolve(
                    m2_ref,
                    Q2m_ref,
                    strong_coupling=strong_coupling,
                    fact_to_ren=theory_dict["fact_to_ren_scale_ratio"] ** 2,
                    q2_to=m2_computed[nf - 3],
                )
            )
        np.testing.assert_allclose(m2_computed, m2_test, rtol=6e-4)

    def test_errors(self):

        # test mass ordering
        with pytest.raises(ValueError, match="Msbar masses are not to be sorted"):
            theory_dict.update(
                dict(
                    mc=1.1,
                    mb=1.0,
                    Qmc=1.2,
                    Qmb=1.0,
                )
            )
            msbar_masses.compute(theory_dict)

        # test forward conditions on alphas_ref
        with pytest.raises(ValueError, match="should be lower than"):
            theory_dict.update(dict(Qmb=91.0001))
            msbar_masses.compute(theory_dict)

        # test backward conditions on alphas_ref
        with pytest.raises(ValueError, match="should be greater than"):
            theory_dict.update(dict(Qmt=89.9999))
            msbar_masses.compute(theory_dict)

        theory_dict.update(dict(Qmb=4.0, Qmt=175))

        # test forward conditions on masses
        with pytest.raises(ValueError, match="should be lower than m"):
            theory_dict.update(dict(mt=174))
            msbar_masses.compute(theory_dict)

        # test backward conditions on masses
        with pytest.raises(ValueError, match="should be greater than m"):
            theory_dict.update(dict(mb=4.1, mt=176))
            msbar_masses.compute(theory_dict)
