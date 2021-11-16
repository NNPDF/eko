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


class TestMsbarMasses:
    def test_compute_msbar_mass(self):
        # Test solution of msbar(m) = m

        theory_dict = {
            "alphas": 0.35,
            "Qref": 1.4,
            "nfref": None,
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
            "Qmc": 1.9,
            "Qmb": 3.9,
            "Qmt": 174.9,
        }
        for method in ["EXA", "EXP"]:
            for order in [1, 2]:
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
                            strong_coupling=strong_coupling,
                            config=dict(
                                fact_to_ren=theory_dict["fact_to_ren_scale_ratio"]
                            ),
                            q2_to=m2_computed[nf - 3],
                        )
                    )
                np.testing.assert_allclose(m2_computed, m2_test, rtol=5e-3)

    def test_errors(self):
        with pytest.raises(ValueError, match="MSBAR"):
            compute_msbar_mass(
                dict(
                    Q0=np.sqrt(0.9),
                    mc=1.0,
                    mb=2.1,
                    mt=3.0,
                    Qmc=1.0,
                    Qmb=2.0,
                    Qmt=3.0,
                    kcThr=1.0,
                    kbThr=1.0,
                    ktThr=1.0,
                    MaxNfPdf=6,
                    HQ="MSBAR",
                    PTO=2,
                    method="EXA",
                    fact_to_ren_scale_ratio=1.0,
                    alphas=0.118,
                    Qref=91 ** 2,
                ),
            )
