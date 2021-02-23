# -*- coding: utf-8 -*-
"""
    Benchmark EKO to Apfel
"""
from banana.data import power_set

from ekomark.benchmark.runner import Runner
from ekomark.data import operators


class ApfelBenchmark(Runner):

    """
    Globally set the external program to Apfel
    """

    external = "apfel"

    # Rotate to evolution basis
    rotate_to_evolution_basis = True

    # pdf to skip
    skip_pdfs = [22, -6, 6, "ph", "T35", "V35"]


class BenchmarkPlain(ApfelBenchmark):
    """Benchmark lo, nlo, scale variation and Intrinsic charm"""

    def benchmark_lo(self):

        theory_updates = {
            "PTO": [0],
            "FNS": ["ZM-VFNS", "FNS"],
            "ModEv": ["EXA", "TRN"],
            "NfFF": [3, 4],
        }
        self.run(
            power_set(theory_updates),
            operators.build(operators.apfel_config),
            ["ToyLH"],
        )

    def benchmark_nlo(self):

        # TODO: other parameter to set as not default?
        theory_updates = {
            "PTO": [1],
            "FNS": ["ZM-VFNS", "FNS"],
            "ModEv": [
                "EXA",
                "EXP",
                "TRN",
                "ordered-truncated",
                "decompose-exact",
                "decompose-expanded",
                "perturbative-exact",
                "perturbative-expanded",
            ],
            # "NfFF": [3, 4],
        }
        self.run(
            power_set(theory_updates),
            operators.build(operators.apfel_config),
            ["ToyLH"],
        )

    def benchmark_sv(self):

        # TODO: other parameter to set as not default?
        theory_updates = {
            "PTO": [1],
            "FNS": ["ZM-VFNS", "FFNS"],
            "NfFF": [4],
            "ModEv": ["EXA",],
            "XIR": [0.7071067811865475, 1.4142135623730951],
        }
        self.run(
            power_set(theory_updates),
            operators.build(operators.apfel_config),
            ["ToyLH"],
        )

    def benchmark_ic(self):

        # TODO: other parameter to set as not default?
        theory_updates = {
            "PTO": [0],
            "FNS": ["ZM-VFNS", "FFNS"],
            "NfFF": [3, 4],
            "ModEv": ["EXA",],
            "IC": [0, 1],
            "mc": [1.4142135623730951, 2.0],
            "Qmc": [1.4142135623730951, 2.0],
        }
        self.run(
            filter(lambda c: c["mc"] == c["Qmc"], power_set(theory_updates)),
            operators.build(operators.apfel_config),
            ["ToyLH"],
        )


if __name__ == "__main__":

    apfel = BenchmarkPlain()
    apfel.benchmark_lo()
    # apfel.benchmark_nlo()
    # apfel.benchmark_sv()
    # apfel.benchmark_ic()
