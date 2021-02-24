# -*- coding: utf-8 -*-
"""
    Benchmark EKO to Apfel
"""
import numpy as np

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
    skip_pdfs = [22, -6, 6, -5, 5, "ph", "T35", "V35"]


class BenchmarkZm(ApfelBenchmark):
    """Benckmark ZM-VFNS """

    zm_theory = {
        "FNS": "ZM-VFNS",
        "ModEv": [
            "EXA",
            "EXP",
            "TRN",
        ],
        "kcThr": 1.0,
        "kbThr": 1.0,
        "ktThr": 1.0,
        "mc": np.sqrt(2.0),
        "mb": 4.5,
        "mt": 173,
    }

    def benchmark_zm(self, pto):

        th = self.zm_theory.copy()
        for key, item in th.items():
            if type(item) != list:
                th[key] = [item]
        th.update({"PTO": [pto]})
        self.run(power_set(th), operators.build(operators.apfel_config), ["ToyLH"])


class BenchmarkFfns(ApfelBenchmark):
    """Benckmark FFNS """

    ffns_theory = {
        "FNS": "FFNS",
        "NfFF": 4,
        "ModEv": [
            "EXA",
            "EXP",
            "TRN",
        ],
        "kcThr": 0.0,
        "kbThr": np.inf,
        "ktThr": np.inf,
    }

    def benchmark_ffns(self, pto):

        th = self.ffns_theory.copy()
        for key, item in th.items():
            if type(item) != list:
                th[key] = [item]
        th.update({"PTO": [pto]})
        self.run(power_set(th), operators.build(operators.apfel_config), ["ToyLH"])

    def benchmark_sv(self):
        """Benckmark Scale Variation"""

        th = self.ffns_theory.copy()
        for key, item in th.items():
            if type(item) != list:
                th[key] = [item]
        th.update({"PTO": [1], "XIR": [0.7071067811865475, 1.4142135623730951]})
        self.run(power_set(th), operators.build(operators.apfel_config), ["ToyLH"])

    def benchmark_ic(self):
        """Benckmark Intrinsic Charm"""

        th = self.ffns_theory.copy()
        for key, item in th.items():
            th[key] = [item]
        th.update(
            {
                "PTO": [1],
                "IC": [1],
                "mc": [1.4142135623730951, 2.0],
                "Qmc": [1.4142135623730951, 2.0],
            }
        )

        self.run(
            filter(lambda c: c["mc"] == c["Qmc"], power_set(th)),
            operators.build(operators.apfel_config),
            ["ToyLH"],
        )


if __name__ == "__main__":

    zm = BenchmarkZm()
    ffns = BenchmarkFfns()
    for o in [0, 1]:
        zm.benchmark_zm(o)
        ffns.benchmark_ffns(o)

    ffns.benchmark_sv()
    ffns.benchmark_ic()
