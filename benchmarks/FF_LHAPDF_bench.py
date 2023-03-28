"""Benchmark FFs from LHAPDF"""

from banana import register

from eko import interpolation
from ekomark.benchmark.runner import Runner

register(__file__)

base_operator = {"ev_op_iterations": 10, "backward_inversion": "exact"}

base_theory = {
    "Qref": 91.1876,
    "mc": 1.51,
    "mb": 4.92,
    "mt": 172.5,
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
    "alphas": 0.118000,
    "alphaqed": 0.007496,
    "FNS": "ZM-VFNS",
    "ModEv": "EXA",
#    "ModEv": "TRN",
}

FF_sets_lo = [
    "NNFF10_PIm_lo",
    "NNFF10_PIp_lo",
    "NNFF10_PIsum_lo",
    "NNFF10_KAm_lo",
    "NNFF10_KAp_lo",
    "NNFF10_KAsum_lo",
    "NNFF10_PRm_lo",
    "NNFF10_PRp_lo",
    "NNFF10_PRsum_lo",
]
FF_sets_nlo = [
    "NNFF10_PIm_nlo",
    "NNFF10_PIp_nlo",
    "NNFF10_PIsum_nlo",
    "NNFF10_KAm_nlo",
    "NNFF10_KAp_nlo",
    "NNFF10_KAsum_nlo",
    "NNFF10_PRm_nlo",
    "NNFF10_PRp_nlo",
    "NNFF10_PRsum_nlo",
    "MAPFF10NLOPIm",
    "MAPFF10NLOPIp",
    "MAPFF10NLOPIsum",
    "MAPFF10NLOKAm",
    "MAPFF10NLOKAp",
    "MAPFF10NLOKAsum",
]
FF_sets_nnlo = [
    "NNFF10_PIm_nnlo",
    "NNFF10_PIp_nnlo",
    "NNFF10_PIsum_nnlo",
    "NNFF10_KAm_nnlo",
    "NNFF10_KAp_nnlo",
    "NNFF10_KAsum_nnlo",
    "NNFF10_PRm_nnlo",
    "NNFF10_PRp_nnlo",
    "NNFF10_PRsum_nnlo",
    "MAPFF10NNLOPIm",
    "MAPFF10NNLOPIp",
    "MAPFF10NNLOPIsum",
    "MAPFF10NNLOKAm",
    "MAPFF10NNLOKAp",
    "MAPFF10NNLOKAsum",
]


class BenchmarkFF(Runner):
    external = "LHAPDF"
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

    def benchmark_lo(self, ff_index, Q0=10, Q2grid=(10000,)):
        theory_card = {
            **base_theory,
            "PTO": 0,
            "QED": 0,
            "Q0": Q0,
            "MaxNfPdf": 5,
            "MaxNfAs": 5,
        }

        operator_card = {
            **base_operator,
            "Q2grid": list(Q2grid),
            "time_like": True,
            "interpolation_xgrid": interpolation.lambertgrid(100, 0.01),
        }

        self.run([theory_card], [operator_card], [FF_sets_lo[ff_index]])

    def benchmark_nlo(self, ff_index, Q0=10, Q2grid=(10000,)):
        theory_card = {
            **base_theory,
            "PTO": 1,
            "QED": 0,
            "Q0": Q0,
            "MaxNfPdf": 5,
            "MaxNfAs": 5,
        }

        operator_card = {
            **base_operator,
            "Q2grid": list(Q2grid),
            "time_like": True,
            "interpolation_xgrid": interpolation.lambertgrid(100, 0.01),
        }

        self.run([theory_card], [operator_card], [FF_sets_nlo[ff_index]])

    def benchmark_nnlo(self, ff_index, Q0=10, Q2grid=(10000,)):
        theory_card = {
            **base_theory,
            "PTO": 2,
            "QED": 0,
            "Q0": Q0,
            "MaxNfPdf": 5,
            "MaxNfAs": 5,
        }

        operator_card = {
            **base_operator,
            "Q2grid": list(Q2grid),
            "time_like": True,
            "interpolation_xgrid": interpolation.lambertgrid(100, 0.01),
        }

        self.run([theory_card], [operator_card], [FF_sets_nnlo[ff_index]])


if __name__ == "__main__":
#    BenchmarkFF().benchmark_lo(7)
    BenchmarkFF().benchmark_nlo(10)
#    BenchmarkFF().benchmark_nnlo(10)
