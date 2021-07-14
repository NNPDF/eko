## -*- coding: utf-8 -*-
"""
Benchmark NNDPF31 pdf family
"""
import numpy as np
from ekomark.benchmark.runner import Runner


class LHAPDFBenchmark(Runner):
    """
    Globally set the external program to LHAPDF
    """

    def __init__(self):
        super().__init__()
        self.external = "LHAPDF"

        # Rotate to evolution basis
        self.rotate_to_evolution_basis = True


base_theory = {
    "Qref": 91.2,
    "mc": 1.51,
    "mb": 4.92,
    "mt": 172.5,
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
    "alphas": 0.118000,
    "FNS": "ZM-VFNS",
}


class BenchmarkNNPDF31(LHAPDFBenchmark):
    """Benchmark NNPDF pdfs"""

    def benchmark_nlo(self, Q0=5, Q2grid=(100,)):
        theory_card = base_theory.copy()
        theory_card.update(
            {
                "PTO": 1,
                "Q0": Q0,
            }
        )
        operator_card = {"Q2grid": list(Q2grid)}
        self.skip_pdfs = lambda _theory: [
            -6,
            6,
            22,
            "ph",
            "T35",
            "V35",
        ]
        self.run([theory_card], [operator_card], ["NNPDF31_nlo_as_0118"])


if __name__ == "__main__":
    lhapdf = BenchmarkNNPDF31()
    low2 = 10
    high2 = 90
    # test forward
    lhapdf.benchmark_nlo(Q0=np.sqrt(low2), Q2grid=[high2])
    # test backward
    lhapdf.benchmark_nlo(Q0=np.sqrt(high2), Q2grid=[low2])
