# -*- coding: utf-8 -*-
"""
Benchmark CT14 pdf family

- as far as I understand, the llo family uses LO splitting functions with LO alphas evolution,
  whereas the lo family uses LO splitting functions with NLO alphas evolution
"""

from banana import register

from ekomark.benchmark.runner import Runner

register(__file__)


base_theory = {
    "Qref": 91.1876,
    "mc": 1.3,
    "mb": 4.75,
    "mt": 172,
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
}


class BenchmarkCT14(Runner):
    """Benchmark CT14 pdfs"""

    external = "LHAPDF"

    # Rotate to evolution basis
    rotate_to_evolution_basis = False

    def benchmark_llo_NF3(self, Q0=5, Q2grid=(100,)):
        theory_card = base_theory.copy()
        theory_card.update(
            {
                "alphas": 0.118000,
                "alphaqed": 0.007496,
                "PTO": 1,
                "QED": 0,
                "Q0": Q0,
                "MaxNfPdf": 3,
                "MaxNfAs": 3,
            }
        )
        operator_card = {"Q2grid": list(Q2grid)}
        self.skip_pdfs = lambda _theory: [
            -6,
            -5,
            -4,
            4,
            5,
            6,
            22,
            "ph",
            "T35",
            "V35",
            "T24",
            "V24",
            "T15",
            "V15",
        ]
        self.run([theory_card], [operator_card], ["CT14llo_NF3"])

    def benchmark_llo_NF4(self, Q0=5, Q2grid=(100,)):
        theory_card = base_theory.copy()
        theory_card.update(
            {
                "alphas": 0.125000,
                "alphaqed": 0.007496,
                "PTO": 1,
                "QED": 0,
                "Q0": Q0,
                "MaxNfPdf": 4,
                "MaxNfAs": 4,
            }
        )
        operator_card = {"Q2grid": list(Q2grid)}
        self.skip_pdfs = lambda _theory: [
            -6,
            -5,
            5,
            6,
            22,
            "ph",
            "T35",
            "V35",
            "T24",
            "V24",
        ]
        self.run([theory_card], [operator_card], ["CT14llo_NF4"])

    def benchmark_llo_NF6(self, Q0=10, Q2grid=(1e6,)):
        theory_card = base_theory.copy()
        theory_card.update(
            {
                "alphas": 0.130000,
                "alphaqed": 0.007496,
                "PTO": 1,
                "QED": 0,
                "Q0": Q0,
                "MaxNfPdf": 6,
                "MaxNfAs": 6,
            }
        )
        operator_card = {"Q2grid": list(Q2grid)}
        self.skip_pdfs = lambda _theory: [22, "ph"]
        self.run([theory_card], [operator_card], ["CT14llo_NF6"])


if __name__ == "__main__":
    b = BenchmarkCT14()
    b.benchmark_llo_NF4()
