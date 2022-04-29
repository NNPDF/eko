# -*- coding: utf-8 -*-
"""
Benchmark CT18 pdf family

"""

from banana import register

from ekomark.benchmark.runner import Runner

register(__file__)


base_theory = {
    "Qref": 91.1870,
    "mc": 1.3,
    "mb": 4.75,
    "mt": 172.0,
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
}


class BenchmarkCT18(Runner):
    """Benchmark CT18 pdfs"""

    external = "LHAPDF"

    # Rotate to evolution basis
    rotate_to_evolution_basis = True

    def benchmark_nnlo(self, Q0=1.295, Q2grid=(1e4,)):
        theory_card = base_theory.copy()
        theory_card.update(
            {
                "alphas": 0.118000,
                "alphaem": 0.00781,
                "PTOs": 2,
                "PTOem": 0,
                "Q0": Q0,
                "MaxNfPdf": 5,
                "MaxNfAs": 5,
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
        self.run([theory_card], [operator_card], ["CT18NNLO"])

    def benchmark_znnlo(self, Q0=1.3, Q2grid=(1e4,)):
        theory_card = base_theory.copy()
        theory_card.update(
            {
                "alphas": 0.118000,
                "alphaem": 0.00781,
                "PTOs": 2,
                "PTOem": 0,
                "Q0": Q0,
                "MaxNfPdf": 5,
                "MaxNfAs": 5,
                "mc": 1.4,
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
        self.run([theory_card], [operator_card], ["CT18ZNNLO"])


if __name__ == "__main__":
    b = BenchmarkCT18()
    b.benchmark_nnlo()
