## -*- coding: utf-8 -*-
"""
    Benchmark EKO to LHAPDF
"""
import pathlib
import yaml

#from banana.data import power_set

from ekomark.benchmark.runner import Runner
#from ekomark.data import operators


class LHAPDFBenchmark(Runner):

    """
    Globally set the external program to LHAPDF
    """

    external = "LHAPDF"

    # Rotate to evolution basis
    # rotate_to_evolution_basis = True
    # TODO: rotate also lhapdf to evbasis


class BenchmarkCT14(LHAPDFBenchmark):
    """Benchmark lo, nlo"""

    def benchmark_lo(self):
        pineko_data = (
            pathlib.Path(__file__).absolute().parents[1] / "data" / "pineko_exercise"
        )
        with open(pineko_data / "theory.yaml") as f:
            theory_card = yaml.safe_load(f)
        with open(pineko_data / "operator.yaml") as f:
            operator_card = yaml.safe_load(f)

        self.run([theory_card], [operator_card], ["CT14llo_NF4"])


if __name__ == "__main__":
    lhapdf = BenchmarkCT14()
    lhapdf.benchmark_lo()
    # lhapdf.benchmark_nlo()

    # TODO: other types of benchmark FNS, sv ??
