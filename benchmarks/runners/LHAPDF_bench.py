## -*- coding: utf-8 -*-
"""
    Benchmark EKO to LHAPDF
"""
import logging
import sys
import os
import pathlib

from banana.data import power_set


from ekomark.benchmark.runner import Runner
from ekomark.data import operators


class LHAPDFBenchmark(Runner):

    """
    Globally set the external program to LHAPDF
    """

    external = "LHAPDF"

    # selcet output type:
    post_process_config = {
        "plot_PDF": True,
        "plot_operator": False,
        "write_operator": False,
    }

    # output dir
    output_path = f"{pathlib.Path(__file__).parents[0]}/{external}_bench"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Rotate to evolution basis
    rtevb = True

    ref = {}


class BenchmarkPlain(LHAPDFBenchmark):
    """The most basic checks"""

    def benchmark_lo(self):

        theory_updates = {
            "PTO": [0],
            "FNS": ["ZM-VFNS", "FNS"],
            "ModEv": ["EXA", "TRN"],
        }
        self.run(
            power_set(theory_updates),
            operators.build(operators.lhapdf_config),
            ["ToyLH"],
        )

    def benchmark_nlo(self):

        # TODO: other parameter to set as not default?
        theory_updates = {
            "PTO": [1],
            "FNS": ["ZM-VFNS", "FNS"],
            "ModEv": [
                "EXA",
                "TRN",
                "ordered-truncated",
                "decompose-exact",
                "decompose-expanded",
                "perturbative-exact",
                "perturbative-expanded",
            ],
        }
        self.run(
            power_set(theory_updates),
            operators.build(operators.lhapdf_config),
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

    lhapdf = BenchmarkPlain()
    lhapdf.benchmark_lo()
    lhapdf.benchmark_nlo()

    # TODO: other types of benchmark FNS, sv ??
