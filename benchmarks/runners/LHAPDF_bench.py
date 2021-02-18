## -*- coding: utf-8 -*-
"""
    Benchmark EKO to LHAPDF
"""
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
    output_path = (
        f"{pathlib.Path(__file__).absolute().parents[1]}/data/{external}_bench"
    )

    # Rotate to evolution basis
    rtevb = True


class BenchmarkPlain(LHAPDFBenchmark):
    """Benchmark lo, nlo"""

    def benchmark_lo(self):

        theory_updates = {
            "PTO": [0],
            "FNS": ["ZM-VFNS", "FNS"],
            "ModEv": ["EXA", "TRN"],
        }
        self.run(
            power_set(theory_updates),
            operators.build(operators.lhapdf_config),
            ["CT14llo_NF4"],
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
            "NfFF": [3, 4],
        }
        self.run(
            power_set(theory_updates),
            operators.build(operators.lhapdf_config),
            ["CT14llo_NF4"],
        )


if __name__ == "__main__":

    lhapdf = BenchmarkPlain()
    lhapdf.benchmark_lo()
    # lhapdf.benchmark_nlo()

    # TODO: other types of benchmark FNS, sv ??
