# -*- coding: utf-8 -*-
"""
    Benchmark EKO to Apfel
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


class ApfelBenchmark(Runner):

    """
    Globally set the external program to Apfel
    """

    external = "apfel"

    # Rotate to evolution basis
    rotate_to_evolution_basis = True

    @staticmethod
    def skip_pdfs(_theory):
        # pdf to skip
        return [22, -6, 6, "ph", "T35", "V35"]


class BenchmarkVFNS(ApfelBenchmark):
    """Benckmark VFNS"""

    vfns_theory = {
        "FNS": "ZM-VFNS",
        "ModEv": [
            "EXA",
            # "EXP",
            # "TRN",
        ],
        "kcThr": 1.0,
        "kbThr": 1.0,
        "ktThr": 1.0,
        "Qref": np.sqrt(2.0),
        "alphas": 0.35,
        "Q0": np.sqrt(2.0),
    }
    vfns_theory = tolist(vfns_theory)

    def benchmark_plain(self, pto):
        """Plain configuration"""

        th = self.vfns_theory.copy()
        th.update({"PTO": [pto]})
        self.run(
            cartesian_product(th), operators.build(operators.apfel_config), ["ToyLH"]
        )

    def benchmark_sv(self, pto):
        """Scale Variation"""

        th = self.vfns_theory.copy()
        th.update({"PTO": [pto], "XIR": [0.7071067811865475, 1.4142135623730951]})
        self.run(
            cartesian_product(th), operators.build(operators.apfel_config), ["ToyLH"]
        )

    def benchmark_kthr(self, pto):
        """Threshold scale different from heavy quark mass"""

        th = self.vfns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "kcThr": [1.23],
                "kbThr": [1.45],
            }
        )
        self.run(
            cartesian_product(th), operators.build(operators.apfel_config), ["ToyLH"]
        )

    def benchmark_msbar(self, pto):
        """MSbar heavy quark mass scheme"""
        # TODO: when there are these settings Apfel use as thr mc*kcThr
        # in eko we are using kcthr*msbar_c where msbar_c is the usual
        # solution.

        th = self.vfns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "kcThr": [1.0],
                "Qmc": [2.6],
                "mc": [2.0],
                "HQ": ["MSBAR"],
            }
        )
        self.run(
            cartesian_product(th), operators.build({"Q2grid": [[10]]}), ["ToyLH"]
        )


class BenchmarkFFNS(ApfelBenchmark):
    """Benckmark FFNS"""

    ffns_theory = {
        "FNS": "FFNS",
        "ModEv": [
            "EXA",
            "EXP",
            "TRN",
        ],
        "NfFF": 4,
        "kcThr": 0.0,
        "kbThr": np.inf,
        "ktThr": np.inf,
        "Q0": 5,
    }
    ffns_theory = tolist(ffns_theory)

    def benchmark_plain(self, pto):
        """Plain configuration"""

        th = self.ffns_theory.copy()
        th.update({"PTO": [pto]})
        self.run(
            cartesian_product(th), operators.build(operators.apfel_config), ["ToyLH"]
        )

    def benchmark_sv(self, pto):
        """Scale Variation"""

        ts = []
        th = self.ffns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "XIR": [np.sqrt(0.5)],
                "fact_to_ren_scale_ratio": [np.sqrt(2.0)],
            }
        )
        ts.extend(cartesian_product(th))
        th = self.ffns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "XIR": [np.sqrt(2.0)],
                "fact_to_ren_scale_ratio": [np.sqrt(0.5)],
            }
        )
        ts.extend(cartesian_product(th))
        self.run(ts, operators.build(operators.apfel_config), ["ToyLH"])


if __name__ == "__main__":

    obj = BenchmarkVFNS()
    # obj = BenchmarkFFNS()

    # obj.benchmark_plain(1)
    # obj.benchmark_sv(1)
    # obj.benchmark_kthr(2)
    obj.benchmark_msbar(2)
