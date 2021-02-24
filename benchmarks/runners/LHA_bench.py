# -*- coding: utf-8 -*-
"""
Benchmark to :cite:`Giele:2002hx`
"""
import numpy as np

from ekomark.benchmark.runner import Runner
from ekomark.data import operators

base_theory = {
    "ModEv": "EXA",
    "Q0": np.sqrt(2.0),  # Eq. (30)
    "mc": np.sqrt(2.0),  # Eq. (34)
    "mb": 4.5,
    "mt": 175,
    "Qref": np.sqrt(2.0),  # Eq. (32)
    "alphas": 0.35,
}
"""Global theory settings"""


class LHABenchmark(Runner):
    """
    Globally set the external program to LHA
    """

    external = "LHA"

    theory = {}

    rotate_to_evolution_basis = True

    skip_pdfs = [22, -6, 6, "ph", "V35", "V24", "V15", "V8", "T35"]

    def plain_theory(self, pto):
        """
        Plain theories at given PTO.

        Parameters
        ----------
            pto : int
                perturbation order

        Returns
        -------
            dict
                theory update
        """
        th = self.theory.copy()
        th.update({"PTO": pto})
        return th

    def sv_theories(self):
        """
        Scale variation theories.

        Returns
        -------
            list(dict)
                theory updates
        """
        low = self.theory.copy()
        low["XIR"] = 0.7071067811865475
        high = self.theory.copy()
        high["XIR"] = 1.4142135623730951
        return [low, high]

    def run_lha(self, theory_updates):
        """
        Enforce operators and PDF

        Parameters
        ----------
            theory_updates : list(dict)
                theory updates
        """
        self.run(theory_updates, operators.build(operators.lha_config), ["ToyLH"])


class BenchmarkVFNS(LHABenchmark):
    """Variable Flavor Number Scheme"""

    theory = base_theory.copy()
    theory.update(
        {
            "FNS": "ZM-VFNS",
            "ModEv": "EXA",
            "kcThr": 1.0,
            "kbThr": 1.0,
            "ktThr": 1.0,
        }
    )

    def benchmark_plain(self, pto):
        """Plain configuration"""
        self.run_lha(self.plain_theory(pto))

    def benchmark_sv(self):
        """Scale variations"""
        self.run_lha(self.sv_theories())


class BenchmarkFFNS(LHABenchmark):
    """Fixed Flavor Number Scheme"""

    theory = base_theory.copy()
    theory.update(
        {
            "FNS": "FFNS",
            "NfFF": 4,
            "kcThr": 0.0,
            "kbThr": np.inf,
            "ktThr": np.inf,
        }
    )

    def benchmark_ffns(self, pto):
        """Plain configuration"""
        self.run_lha(self.plain_theory(pto))

    def benchmark_sv(self):
        """Scale Variation"""
        self.run_lha(self.sv_theories())


if __name__ == "__main__":

    zm = BenchmarkVFNS()
    ffns = BenchmarkFFNS()
    # for o in [1]:
    #    zm.benchmark_zm(o)
    #    ffns.benchmark_ffns(o)

    zm.benchmark_sv()
    # ffns.benchmark_sv()
