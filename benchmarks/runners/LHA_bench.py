# -*- coding: utf-8 -*-
"""
    Benchmark EKO to :cite:`Giele:2002hx`
"""
import numpy as np

from banana.data import power_set

from ekomark.benchmark.runner import Runner
from ekomark.data import operators


ffns_theory = {
    "FNS": "FFNS",
    "NfFF": 4,
    "ModEv": "EXA",
    "Q0": np.sqrt(2),
    "kcThr": 0.0,
    "kbThr": np.inf,
    "ktThr": np.inf,
    "Qref": np.sqrt(2.0),
    "alphas": 0.35,
}

zm_theory = {
    "FNS": "ZM-VFNS",
    "ModEv": "EXA",
    "Q0": np.sqrt(2),
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
    "mc": np.sqrt(2.0),
    "mb": 4.5,
    "mt": 175,
    "Qref": np.sqrt(2.0),
    "alphas": 0.35,
}


class LHABenchmark(Runner):

    """
    Globally set the external program to LHA
    """

    external = "LHA"

    rotate_to_evolution_basis = True

    # pdf to skip
    skip_pdfs = [22, -6, 6, "ph", "V35", "V24", "V15", "V8", "T35"]


class BenchmarkPlain(LHABenchmark):
    """Vary PTO and scale variations """

    def benchmark_lo(self):

        lo = zm_theory.copy()
        lo.update({"PTO": 0})

        self.run([lo], operators.build(operators.lha_config), ["ToyLH"])

    def benchmark_nlo(self):

        theory_updates = {
            "PTO": [1],
            "FNS": ["ZM-VFNS", "FFNS"],
            "ModEv": [
                "EXA",
                # "TRN",
                # "ordered-truncated",
                # "decompose-exact",
                # "decompose-expanded",
                # "perturbative-exact",
                # "perturbative-expanded",
            ],
        }
        self.run(
            power_set(theory_updates), operators.build(operators.lha_config), ["ToyLH"]
        )

    def benchmark_sv(self):

        theory_updates = {
            "PTO": [1],
            # "FNS": ["ZM-VFNS", "FFNS"],
            "ModEv": ["EXA",],
            "XIR": [0.7071067811865475, 1.4142135623730951],
        }
        self.run(
            power_set(theory_updates), operators.build(operators.lha_config), ["ToyLH"]
        )


if __name__ == "__main__":

    lha = BenchmarkPlain()
    lha.benchmark_lo()
    # lha.benchmark_nlo()
    # lha.benchmark_sv()
