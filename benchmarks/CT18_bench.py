"""
Benchmark CT18 pdf family

"""

from banana import register

from ekomark.benchmark.runner import Runner

register(__file__)


base_theory = {
    "Qref": 91.1870,
    "Qedref": 91.1870,
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

    def benchmark_nnlo(self, Q0=1.295, mugrid=(1e4,)):
        theory_card = base_theory.copy()
        theory_card.update(
            {
                "alphas": 0.118000,
                "alphaqed": 0.007496,
                "PTO": 2,
                "QED": 0,
                "Q0": Q0,
                "MaxNfPdf": 5,
                "MaxNfAs": 5,
            }
        )
        operator_card = {"mugrid": list(mugrid)}
        self.skip_pdfs = lambda _theory: [
            -6,
            6,
            22,
            "ph",
            "T35",
            "V35",
        ]
        self.run([theory_card], [operator_card], ["CT18NNLO"])

    def benchmark_nnlo_qed(self, Q0=1.295, mugrid=(1e4,)):
        theory_card = base_theory.copy()
        theory_card.update(
            {
                "alphas": 0.118000,
                "alphaqed": 0.007496,
                "PTO": 2,
                "QED": 1,
                "Q0": Q0,
                "MaxNfPdf": 5,
                "MaxNfAs": 5,
            }
        )
        operator_card = {"mugrid": list(mugrid)}
        self.skip_pdfs = lambda _theory: [
            -6,
            6,
            "Tu8",
            "Vu8",
        ]
        self.run([theory_card], [operator_card], ["CT18qed"])

    def benchmark_znnlo(self, Q0=1.3, mugrid=(1e4,)):
        theory_card = base_theory.copy()
        theory_card.update(
            {
                "alphas": 0.118000,
                "alphaqed": 0.007496,
                "PTO": 2,
                "QED": 0,
                "Q0": Q0,
                "MaxNfPdf": 5,
                "MaxNfAs": 5,
                "mc": 1.4,
            }
        )
        operator_card = {"mugrid": list(mugrid)}
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
