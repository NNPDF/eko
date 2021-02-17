# -*- coding: utf-8 -*-
"""
    Benchmark EKO to Apfel
"""
import logging
import sys
import os
import pathlib

from banana.data import power_set


from ekomark.benchmark.runner import Runner
from ekomark.data import operators


class ApfelBenchmark(Runner):

    """
    Globally set the external program to Apfel
    """

    external = "apfel"

    # selcet output type:
    post_process_config = {
        "plot_PDF": True,
        "plot_operator": False,
        "write_operator": False,
    }

    # output dir
    output_path = (
        f"{pathlib.Path(__file__).absolute().parents[1]}/data/{external}_bench"
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Rotate to evolution basis
    rtevb = True


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
            filter(lambda c: c["mc"] == c["Qmc"], power_set(sv)),
            operators.build(operators.apfel_config),
            ["ToyLH"],
        )


if __name__ == "__main__":

    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko").handlers = []
    logging.getLogger("eko").addHandler(logStdout)
    logging.getLogger("eko").setLevel(logging.INFO)

    apfel = BenchmarkPlain()
    apfel.benchmark_lo()
    # apfel.benchmark_nlo()
    # apfel.benchmark_sv()
    # apfel.benchmark_ic()
