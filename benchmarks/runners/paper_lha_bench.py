# -*- coding: utf-8 -*-
"""
Benchmark to :cite:`Giele:2002hx`
"""
import numpy as np

from ekomark.benchmark.runner import Runner

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
            list(dict)
                theory updates
        """
        th = self.theory.copy()
        th.update({"PTO": pto})
        return [th]

    def sv_theories(self):
        """
        Scale variation theories.

        Returns
        -------
            list(dict)
                theory updates
        """
        low = self.theory.copy()
        low["PTO"] = 1
        low["XIR"] = np.sqrt(1.0 / 2.0)
        high = self.theory.copy()
        high["PTO"] = 1
        high["XIR"] = np.sqrt(2.0)
        return [low, high]

    def run_lha(self, theory_updates):
        """
        Enforce operators and PDF

        Parameters
        ----------
            theory_updates : list(dict)
                theory updates
        """
        self.run(theory_updates, [{"Q2grid": [1e4],"ev_op_iterations": 10,}], ["ToyLH"])

    def benchmark_plain(self, pto):
        """Plain configuration"""
        self.run_lha(self.plain_theory(pto))

    def benchmark_sv(self):
        """Scale variations"""
        self.run_lha(self.sv_theories())


class BenchmarkVFNS(LHABenchmark):
    """Variable Flavor Number Scheme"""

    theory = base_theory.copy()
    theory.update(
        {
            "FNS": "ZM-VFNS",  # ignored by eko, but needed by LHA_utils
            "ModEv": "EXA",
            "kcThr": 1.0,
            "kbThr": 1.0,
            "ktThr": 1.0,
        }
    )


class BenchmarkFFNS(LHABenchmark):
    """Fixed Flavor Number Scheme"""

    theory = base_theory.copy()
    theory.update(
        {
            "FNS": "FFNS",  # ignored by eko, but needed by LHA_utils
            "NfFF": 4,
            "kcThr": 0.0,
            "kbThr": np.inf,
            "ktThr": np.inf,
        }
    )


if __name__ == "__main__":

    vfns = BenchmarkVFNS()
    ffns = BenchmarkFFNS()

    # vfns.benchmark_plain(1)
    ffns.benchmark_plain(1)
    # vfns.benchmark_sv()
    # ffns.benchmark_sv()
