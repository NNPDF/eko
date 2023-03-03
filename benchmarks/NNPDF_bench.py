"""
Benchmark NNPDF pdf family
"""
import numpy as np
from banana import register

from eko import interpolation
from ekomark.benchmark.runner import Runner

register(__file__)


class BenchmarkNNPDF(Runner):
    """
    Globally set the external program to LHAPDF
    """

    external = "LHAPDF"

    # Rotate to evolution basis
    rotate_to_evolution_basis = True

    def skip_pdfs(self, _theory):
        return [
            -6,
            6,
            22,
            "ph",
            "T35",
            "V35",
        ]


base_operator = {"ev_op_iterations": 1, "backward_inversion": "exact"}

base_theory = {
    "Qref": 91.2,
    "mc": 1.51,
    "mb": 4.92,
    "mt": 172.5,
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
    "alphas": 0.118000,
    "alphaqed": 0.007496,
    "FNS": "ZM-VFNS",
    "ModEv": "TRN",
}


class BenchmarkNNPDF31(BenchmarkNNPDF):
    """Benchmark NNPDF3.1"""

    def benchmark_nlo(self, Q0=1.65, Q2grid=(100,)):
        theory_card = {
            **base_theory,
            "PTO": 1,
            "QED": 0,
            "Q0": Q0,
        }

        operator_card = {**base_operator, "Q2grid": list(Q2grid)}
        self.run([theory_card], [operator_card], ["NNPDF31_nlo_as_0118"])


class BenchmarkNNPDF31_luxqed(BenchmarkNNPDF):
    """Benchmark NNPDF3.1_luxqed"""

    def benchmark_nnlo(self, Q0=1.65, Q2grid=(100,)):
        theory_card = {
            **base_theory,
            "PTO": 2,
            "QED": 2,
            "Q0": Q0,
        }
        theory_card.update({"FNS": "VFNS", "QrefQED": 91.2})

        self.skip_pdfs = lambda _theory: [
            -6,
            6,
            "Tu8",
            "Vu8",
        ]

        operator_card = {
            **base_operator,
            "Q2grid": list(Q2grid),
            "ev_op_iterations": 10,
        }
        self.run([theory_card], [operator_card], ["NNPDF31_nnlo_as_0118_luxqed"])


class BenchmarkNNPDF40(BenchmarkNNPDF):
    """Benchmark NNPDF4.0"""

    def benchmark_nlo(self, Q0=1.65, Q2grid=(100,)):
        theory_card = {
            **base_theory,
            "PTO": 1,
            "QED": 0,
            "Q0": Q0,
        }

        operator_card = {**base_operator, "Q2grid": list(Q2grid)}
        self.run([theory_card], [operator_card], ["NNPDF40_nlo_as_01180"])

    def benchmark_nnlo(self, Q0=1.65, Q2grid=(100,)):
        theory_card = {
            **base_theory,
            "PTO": 2,
            "QED": 0,
            "IC": 1,
            "IB": 1,
            "Q0": Q0,
        }

        operator_card = {**base_operator, "Q2grid": list(Q2grid)}
        self.run([theory_card], [operator_card], ["NNPDF40_nnlo_as_01180"])


class BenchmarkNNPDFpol11(BenchmarkNNPDF):
    """Benchmark NNPDFpol11"""

    def benchmark(self, Q0=1.65, Q2grid=(100,)):
        theory_card = {
            "Qref": 91.2,
            "mc": 1.41421,
            "mb": 4.75,
            "mt": 175,
            "kcThr": 1.0,
            "kbThr": 1.0,
            "ktThr": 1.0,
            "alphas": 0.119002,
            "alphaqed": 0.007496,
            "FNS": "ZM-VFNS",
            "ModEv": "TRN",
            "Q0": Q0,
            "PTO": 1,
        }

        operator_card = {
            **base_operator,
            "Q2grid": list(Q2grid),
            "polarized": [True],
            "interpolation_xgrid": interpolation.lambertgrid(50, 1e-5),
        }
        self.run([theory_card], [operator_card], ["NNPDFpol11_100"])


if __name__ == "__main__":
    low2 = 5**2
    high2 = 30**2
    # nn31 = BenchmarkNNPDF31()
    # # test forward
    # nn31.benchmark_nlo(Q0=np.sqrt(low2), Q2grid=[10])
    # # test backward
    # #nn31.benchmark_nlo(Q0=np.sqrt(high2), Q2grid=[low2])
    # nn40 = BenchmarkNNPDF40()
    # # nn40.benchmark_nnlo(Q2grid=[100])
    # nn40.benchmark_nnlo(Q0=np.sqrt(high2), Q2grid=[low2])
    # nnpol = BenchmarkNNPDFpol11()
    # nnpol.benchmark(Q0=np.sqrt(low2), Q2grid=[high2])
    obj = BenchmarkNNPDF31_luxqed()
    obj.benchmark_nnlo(Q0=5.0)
