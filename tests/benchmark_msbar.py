# -*- coding: utf-8 -*-
"""This module benchmarks MSbar mass evolution against APFEL."""
import numpy as np

from eko.msbar_masses import evolve_msbar_mass
from eko.strong_coupling import StrongCoupling


# try to load APFEL - if not available, we'll use the cached values
try:
    import apfel

    use_APFEL = True
except ImportError:
    use_APFEL = False


class BenchmarkMSbar:
    def benchmark_APFEL_msbar(self):
        Q2s = np.power([30, 96, 150], 2)
        alphas_ref = 0.118
        scale_ref = 91.0 ** 2
        thresholds_ratios = np.power((1.0, 1.0, 1.0), 2)
        Q2m = np.power([2.0, 4.5, 175], 2)
        m2 = np.power((1.4, 4.5, 175), 2)
        apfel_vals_dict = {
            0: np.array(
                [
                    [1.009776631306431, 3.665411704463702, 200.84989203003173],
                    [0.9184216270407894, 3.3338000474743867, 182.67890037277968],
                    [0.8892735505271812, 3.2279947658871553, 176.88119438599423],
                ]
            ),
            1: np.array(
                [
                    [0.9175075173036495, 3.4953909932543445, 204.79548921610655],
                    [0.8227578662769597, 3.134427109507681, 183.6465604399247],
                    [0.7934449726444709, 3.0227549733595973, 177.10367302092334],
                ]
            ),
            2: np.array(
                [
                    [0.9019843316830822, 3.4760144608237042, 205.17758283939273],
                    [0.8078493428925942, 3.113242546931765, 183.76436225206226],
                    [0.7788276351007536, 3.0014003869088755, 177.16269119698978],
                ]
            ),
        }
        # collect my values
        for order in [0, 1, 2]:
            as_VFNS = StrongCoupling(
                alphas_ref,
                scale_ref,
                m2,
                thresholds_ratios,
                order=order,
                method="expanded",
                hqm_scheme="MSBAR",
                q2m_ref=Q2m,
            )
            my_vals = []
            for Q2 in Q2s:
                my_masses = []
                for n in [3, 4, 5]:
                    my_masses.append(
                        evolve_msbar_mass(
                            m2[n - 3],
                            Q2m[n - 3],
                            as_VFNS,
                            fact_to_ren=1,
                            nf=n,
                            q2_to=Q2,
                        )
                    )
                my_vals.append(my_masses)
            # get APFEL numbers - if available else use cache
            apfel_vals = apfel_vals_dict[order]
            if use_APFEL:
                # run apfel
                apfel.CleanUp()
                apfel.SetTheory("QCD")
                apfel.SetPerturbativeOrder(order)
                apfel.SetAlphaEvolution("exact")
                apfel.SetAlphaQCDRef(alphas_ref, np.sqrt(scale_ref))
                apfel.SetVFNS()
                apfel.SetMSbarMasses(*np.sqrt(m2))
                apfel.SetMassScaleReference(*np.sqrt(Q2m))
                apfel.SetRenFacRatio(1)
                apfel.InitializeAPFEL()
                # collect a_s
                apfel_vals_cur = []
                for Q2 in Q2s:
                    masses = []
                    for n in [4, 5, 6]:
                        masses.append(apfel.HeavyQuarkMass(n, np.sqrt(Q2)))
                    apfel_vals_cur.append(masses)
                print(apfel_vals_cur)
                np.testing.assert_allclose(apfel_vals, np.array(apfel_vals_cur))
            # check myself to APFEL
            # TODO: looks not so precise ..., pass more physical Q2s
            np.testing.assert_allclose(
                apfel_vals, np.sqrt(np.array(my_vals)), rtol=8e-2
            )
