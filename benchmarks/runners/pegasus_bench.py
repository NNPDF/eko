# -*- coding: utf-8 -*-
"""
Benchmark to Pegasus :cite:`Vogt:2004ns`
"""
import numpy as np

from banana.data import cartesian_product

from ekomark.benchmark.runner import Runner
from ekomark.data import operators


def tolist(input_dict):
    output_dict = input_dict.copy()
    for key, item in output_dict.items():
        if not isinstance(item, list):
            output_dict[key] = [item]
    return output_dict


class PegasusBenchmark(Runner):

    """
    Globally set the external program to Pegasus
    """

    external = "pegasus"

    # Rotate to evolution basis
    rotate_to_evolution_basis = True

    @staticmethod
    def skip_pdfs(_theory):
        # pdf to skip
        return [22, -6, 6, "ph", "T35", "V35"]


class BenchmarkZM(PegasusBenchmark):
    """Benckmark ZM-VFNS """

    zm_theory = {
        "FNS": "ZM-VFNS",
        "ModEv": [
            "EXA",
            "EXP",
            "ordered-truncated",
        ],
        "kcThr": 1.0,
        "kbThr": 1.0,
        "ktThr": 1.0,
        "Qref": np.sqrt(2.0),
        "alphas": 0.35,
        "Q0": np.sqrt(2.0),
    }
    zm_theory = tolist(zm_theory)

    def benchmark_zm(self, pto=1):
        """Benckmark ZM-VFNS LO, NLO and NNLO """

        th = self.zm_theory.copy()
        th.update(
            {
                "PTO": [pto],
            }
        )
        self.run(
            cartesian_product(th), operators.build(operators.pegasus_config), ["ToyLH"]
        )

    def benchmark_sv(self):
        """Benckmark Scale Variation"""

        th = self.zm_theory.copy()
        th.update(
            {
                "PTO": [1, 2],
                "XIR": [0.7071067811865475, 1.4142135623730951],
            }
        )
        self.run(
            cartesian_product(th), operators.build(operators.pegasus_config), ["ToyLH"]
        )


class BenchmarkFFNS(PegasusBenchmark):
    """Benckmark FFNS """

    ffns_theory = {
        "FNS": "FFNS",
        "ModEv": [
            "EXA",
            "EXP",
            "ordered-truncated",
        ],
        "NfFF": 4,
        "kcThr": 0.0,
        "kbThr": np.inf,
        "ktThr": np.inf,
        "Qref": np.sqrt(2.0),
        "alphas": 0.35,
        "Q0": np.sqrt(2.0),
    }
    ffns_theory = tolist(ffns_theory)

    def benchmark_ffns(self, pto=1):
        """Benckmark FFNS LO, NLO and NNLO """

        th = self.ffns_theory.copy()
        th.update(
            {
                "PTO": [pto],
            }
        )
        self.run(
            cartesian_product(th), operators.build(operators.pegasus_config), ["ToyLH"]
        )

    def benchmark_sv(self):
        """Benckmark Scale Variation"""

        th = self.ffns_theory.copy()
        th.update(
            {
                "PTO": [1, 2],
                "XIR": [0.7071067811865475, 1.4142135623730951],
            }
        )
        self.run(
            cartesian_product(th), operators.build(operators.pegasus_config), ["ToyLH"]
        )


if __name__ == "__main__":

    zm = BenchmarkZM()
    ffns = BenchmarkFFNS()
    for o in [0, 1, 2]:
        zm.benchmark_zm(o)
        ffns.benchmark_ffns(o)

    ffns.benchmark_sv()
    zm.benchmark_sv()
