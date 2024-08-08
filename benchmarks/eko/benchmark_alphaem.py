"""This module benchmarks alpha_em against alphaQED23 and validphys.

alphaQED23 can be obtained from http://www-com.physik.hu-berlin.de/~fjeger/software.html .
"""

import numpy as np
import pytest

from eko.couplings import Couplings
from eko.quantities.couplings import CouplingEvolutionMethod, CouplingsInfo
from eko.quantities.heavy_quarks import QuarkMassScheme


@pytest.mark.isolated
class BenchmarkCouplings:
    def test_alphaQED_high(self):
        """testing aginst alphaQED23 for high Q values"""
        alphaQED23 = np.array(
            [
                0.0075683,
                0.0075867,
                0.0076054,
                0.0076245,
                0.0076437,
                0.0076631,
                0.0076827,
                0.0077024,
                0.0077222,
                0.0077421,
                0.0077621,
            ]
        )
        qvec = np.geomspace(10, 100, 11)
        couplings = CouplingsInfo.from_dict(
            dict(
                alphas=0.118,
                alphaem=7.7553e-03,
                ref=(91.2, 5),
                em_running=True,
            )
        )
        eko_alpha = Couplings(
            couplings,
            (3, 2),
            method=CouplingEvolutionMethod.EXACT,
            masses=[m**2 for m in [1.51, 4.92, 172.5]],
            hqm_scheme=QuarkMassScheme.POLE,
            thresholds_ratios=[1.0, 1.0, np.inf],
        )
        alpha_eko = np.array([eko_alpha.a_em(q**2) * 4 * np.pi for q in qvec])

        np.testing.assert_allclose(alphaQED23, alpha_eko, rtol=1.8e-4)

    def test_alphaQED_low(self):
        """testing aginst alphaQED23 for low Q values: they are close but not identical"""
        alphaQED23 = np.array(
            [
                0.0074192,
                0.007431,
                0.0074434,
                0.0074565,
                0.0074702,
                0.0074847,
                0.0075,
                0.0075161,
                0.0075329,
                0.0075503,
                0.0075683,
            ]
        )
        qvec = np.geomspace(1, 10, 11)
        couplings = CouplingsInfo.from_dict(
            dict(
                alphas=0.118,
                alphaem=7.7553e-03,
                ref=(91.2, 5),
                em_running=True,
            )
        )
        eko_alpha = Couplings(
            couplings,
            (3, 2),
            method=CouplingEvolutionMethod.EXACT,
            masses=[m**2 for m in [1.51, 4.92, 172.5]],
            hqm_scheme=QuarkMassScheme.POLE,
            thresholds_ratios=[1.0, 1.0, np.inf],
        )
        alpha_eko = np.array([eko_alpha.a_em(q**2) * 4 * np.pi for q in qvec])

        np.testing.assert_allclose(alphaQED23, alpha_eko, rtol=3.2e-3)

    def test_validphys(self):
        """testing aginst validphys"""
        alpha_vp = np.array(
            [
                0.007408255774054356,
                0.007425240094018394,
                0.007449051094996458,
                0.007476301027742958,
                0.007503751810862984,
                0.007532299008699658,
                0.007561621958607614,
                0.007591174885612722,
                0.007620960508577136,
                0.00765098158940323,
                0.007681240933888789,
                0.007711741392602655,
                0.007742485861781425,
                0.007773477284247778,
                0.007804718650351058,
                0.007836212998930739,
                0.00786796341830342,
                0.007899973047274033,
                0.007932245076171957,
                0.00796478274791273,
            ]
        )
        qvec = np.geomspace(1, 1000, 20)
        couplings = CouplingsInfo.from_dict(
            dict(
                alphas=0.118,
                alphaem=7.7553e-03,
                ref=(91.2, 5),
                em_running=True,
            )
        )
        eko_alpha = Couplings(
            couplings,
            (3, 2),
            method=CouplingEvolutionMethod.EXACT,
            masses=[m**2 for m in [1.51, 4.92, 172.5]],
            hqm_scheme=QuarkMassScheme.POLE,
            thresholds_ratios=[1.0, 1.0, np.inf],
        )
        eko_alpha.decoupled_running = True
        alpha_eko = np.array([eko_alpha.a_em(q**2) * 4 * np.pi for q in qvec])

        np.testing.assert_allclose(alpha_vp, alpha_eko, rtol=5e-6)
