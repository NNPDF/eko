"""Benchmark DSSV pdf family.

Note that the PDF set is private, but can be obtained from the authors
upon request.
"""

from banana import register

from eko import interpolation
from ekomark.benchmark.runner import Runner

register(__file__)


class BenchmarkDSSV(Runner):
    """Benchmark DSSV pdf family."""

    external = "LHAPDF"

    # Rotate to evolution basis
    rotate_to_evolution_basis = True

    def skip_pdfs(self, _theory):
        return [-6, 6, 5, -5, 4, -4, 22, "ph", "T35", "V35", "T24", "V24", "T15", "V15"]

    def benchmark(self, Q0=1.65, mugrid=(100,)):
        theory_card = {
            "Qref": 1.0,
            "mc": 1.4,
            "mb": 4.75,
            "mt": 175,
            "kcThr": 1.0,
            "kbThr": 1.0,
            "ktThr": 1.0,
            "alphas": 0.49127999999999999,
            "FNS": "ZM-VFNS",
            "ModEv": "EXA",
            "Q0": Q0,
            "PTO": 1,
            "MaxNfPdf": 3,
            "MaxNfAs": 3,
        }

        operator_card = {
            "mugrid": list(mugrid),
            "polarized": [True],
            "interpolation_xgrid": interpolation.lambertgrid(50, 1e-5),
        }
        self.run([theory_card], [operator_card], ["DSSV_REP_LHAPDF6"])


if __name__ == "__main__":
    low = 5
    high = 30
    dssv = BenchmarkDSSV()
    dssv.benchmark(Q0=low, mugrid=[high])
