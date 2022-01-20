## -*- coding: utf-8 -*-
"""
Benchmark HERAPDF2.0 pdf family

"""

from ekomark import register
from ekomark.benchmark.runner import Runner

register(__file__)


base_theory = {
    "Qref": 91.1876,
    "mc": 1.43,
    "mb": 4.5,
    "mt": 173.0,
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
}


class BenchmarkHERA20(Runner):
    """Benchmark HERA20 pdfs"""

    external = "LHAPDF"

    # Rotate to evolution basis
    rotate_to_evolution_basis = True

    def benchmark_nnlo(self, Q0=1.3, Q2grid=(1e4,)):
        theory_card = base_theory.copy()
        theory_card.update(
            {
                "alphas": 0.118000,
                "PTO": 2,
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
        self.run([theory_card], [operator_card], ["HERAPDF20_NNLO_EIG"])


if __name__ == "__main__":
    b = BenchmarkHERA20()
    b.benchmark_nnlo()
