# -*- coding: utf-8 -*-
"""
    Benchmark EKO to Apfel
"""
import numpy as np
from banana import register
from banana.data import cartesian_product

from ekomark.benchmark.runner import Runner
from ekomark.data import operators

register(__file__)


def tolist(input_dict):
    output_dict = input_dict.copy()
    for key, item in output_dict.items():
        if not isinstance(item, list):
            output_dict[key] = [item]
    return output_dict


class ApfelBenchmark(Runner):

    """
    Globally set the external program to Apfel
    """

    external = "apfel"

    # Rotate to evolution basis
    rotate_to_evolution_basis = False

    @staticmethod
    def skip_pdfs(_theory):
        # pdf to skip
        return [-6, 6, "Tu8", "Vu8"]


class BenchmarkFFNS(ApfelBenchmark):
    """Benckmark FFNS"""

    ffns_theory = {
        "FNS": "FFNS",
        "ModEv": [
            "EXA",
            # "EXP",
            # "TRN",
        ],
        "NfFF": 5,
        "kcThr": 0.0,
        "kbThr": 0.0,
        "ktThr": np.inf,
        "Q0": 5.0,
    }
    ffns_theory = tolist(ffns_theory)

    def benchmark_plain(self, pto, qed):
        """Plain configuration"""

        th = self.ffns_theory.copy()
        th.update({"PTO": [pto], "QED": [qed]})
        self.run(
            cartesian_product(th),
            operators.build(operators.apfel_config),
            ["CT14qed_proton"],
        )


if __name__ == "__main__":

    obj = BenchmarkFFNS()

    obj.benchmark_plain(1, 1)
