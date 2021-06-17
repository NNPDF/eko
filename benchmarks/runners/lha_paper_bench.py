# -*- coding: utf-8 -*-
"""
Benchmark to :cite:`Giele:2002hx` (LO + NLO) and :cite:`Dittmar:2005ed` (NNLO)
"""
import numpy as np

from ekomark.benchmark.runner import Runner

base_theory = {
    "ModEv": "EXA",
    "Q0": np.sqrt(
        2.0
    ),  # Eq. (30) :cite:`Giele:2002hx`, Eq. (4.53) :cite:`Dittmar:2005ed`
    "nfref": 3,
    "mc": np.sqrt(
        2.0
    ),  # Eq. (34) :cite:`Giele:2002hx`, Eq. (4.56) :cite:`Dittmar:2005ed`
    "mb": 4.5,
    "mt": 175,
    "Qref": np.sqrt(
        2.0
    ),  # Eq. (32) :cite:`Giele:2002hx`,Eq. (4.53) :cite:`Dittmar:2005ed`
    "alphas": 0.35,  # Eq. (4.55) :cite:`Dittmar:2005ed`
}
"""Global theory settings"""

default_skip_pdfs = [22, -6, 6, "ph", "V35", "V24", "V15", "V8", "T35"]
# ffns_skip_pdfs = vfns_skip_pdfs.copy()
# ffns_skip_pdfs.extend([-5, 5, "T24"])


class LHABenchmark(Runner):
    """
    Globally set the external program to LHA
    """

    external = "LHA"

    theory = {}

    rotate_to_evolution_basis = True

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

    def sv_theories(self, pto):
        """
        Scale variation theories.

        Parameters
        ----------
            pto : int
                perturbation order

        Returns
        -------
            list(dict)
                theory updates
        """
        low = self.theory.copy()
        low["PTO"] = pto
        low["fact_to_ren_scale_ratio"] = np.sqrt(1.0 / 2.0)
        high = self.theory.copy()
        high["PTO"] = pto
        high["fact_to_ren_scale_ratio"] = np.sqrt(2.0)
        return [high, low]

    @staticmethod
    def skip_pdfs(_theory):
        """
        Adjust skip_pdf by the used theory

        Parameters
        ----------
            theory_updates : list(dict)
                theory updates

        Returns
        -------
            list :
                current skip_pdf
        """
        return default_skip_pdfs

    def run_lha(self, theory_updates):
        """
        Enforce operators and PDF

        Parameters
        ----------
            theory_updates : list(dict)
                theory updates
        """
        self.run(
            theory_updates,
            [
                {
                    "Q2grid": [1e4],
                    "ev_op_iterations": 10,
                    # "debug_skip_singlet": True
                }
            ],
            ["ToyLH"],
        )

    def benchmark_plain(self, pto):
        """Plain configuration"""
        self.run_lha(self.plain_theory(pto))

    def benchmark_sv(self, pto):
        """Scale variations"""
        self.run_lha(self.sv_theories(pto))


class BenchmarkVFNS(LHABenchmark):
    """Variable Flavor Number Scheme"""

    theory = base_theory.copy()
    theory.update(
        {
            "FNS": "ZM-VFNS",  # ignored by eko, but needed by LHA_utils
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

    @staticmethod
    def skip_pdfs(theory):
        ffns_skip_pdfs = default_skip_pdfs.copy()
        # remove bottom
        ffns_skip_pdfs.extend([-5, 5, "T24"])
        # in NNLO V8 becomes available
        if theory["PTO"] >= 2:
            ffns_skip_pdfs.remove("V8")
        return ffns_skip_pdfs


if __name__ == "__main__":

    obj = BenchmarkVFNS()
    #obj = BenchmarkFFNS()

    obj.benchmark_plain(1)
    #obj.benchmark_sv(1)
