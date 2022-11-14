# -*- coding: utf-8 -*-
"""
    Benchmark EKO to Apfel
"""
import numpy as np
from banana import register
from banana.data import cartesian_product

from ekomark.benchmark.runner import Runner
from ekomark.data import operators

register(__file__)


def tolist(input_dict):
    output_dict = input_dict.copy()
    for key, item in output_dict.items():
        if not isinstance(item, list):
            output_dict[key] = [item]
    return output_dict


class ApfelBenchmark(Runner):

    """
    Globally set the external program to Apfel
    """

    external = "apfel"

    # Rotate to evolution basis
    rotate_to_evolution_basis = True

    @staticmethod
    def skip_pdfs(_theory):
        # pdf to skip
        return [22, -6, 6, "ph", "T35", "V35"]


class BenchmarkVFNS(ApfelBenchmark):
    """Benchmark VFNS"""

    vfns_theory = {
        "FNS": "ZM-VFNS",
        "ModEv": [
            "EXA",
            "EXP",
            "TRN",
        ],
        "kcThr": 1.0,
        "kbThr": 1.0,
        "ktThr": 1.0,
        "Qref": np.sqrt(2.0),
        "alphas": 0.35,
        "Q0": np.sqrt(2.0),
        "nfref": 3,
        "nf0": 3,
        "mc": 1.51,
    }
    vfns_theory = tolist(vfns_theory)

    def benchmark_plain(self, pto):
        """Plain configuration"""

        th = self.vfns_theory.copy()
        th.update({"PTO": [pto]})
        self.run(
            cartesian_product(th), operators.build(operators.apfel_config), ["ToyLH"]
        )

    def benchmark_sv(self, pto, svmode):
        """Scale Variation"""

        th = self.vfns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "XIR": [1 / np.sqrt(2.0)],
                "fact_to_ren_scale_ratio": [np.sqrt(2.0)],
                "ModSV": [svmode],
                "EScaleVar": [0],
                "nfref": [4],
            }
        )
        self.run(
            cartesian_product(th), operators.build(operators.apfel_config), ["ToyLH"]
        )

    def benchmark_kthr(self, pto):
        """Threshold scale different from heavy quark mass"""

        th = self.vfns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "kcThr": [1.23],
                "kbThr": [1.45],
            }
        )
        self.run(
            cartesian_product(th), operators.build(operators.apfel_config), ["ToyLH"]
        )

    def benchmark_msbar(self, pto):
        """
        MSbar heavy quark mass scheme
        when  passing kthr != 1 both apfel and eko use ``kThr * msbar``,
        as thr scale, where ``msbar`` is the usual ms_bar solution.
        However apfel and eko manage the alpha_s thr differently:
        apfel uses the given mass parameters as thr multiplied by the kthr,
        while in eko only the already computed thr matters, so the
        benchmark is not a proper comparison with this option allowed.
        """
        th = self.vfns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                # "kcThr": [1.2],
                # "kbThr": [1.8],
                "Qmc": [18],
                "mc": [1.5],
                "Qmb": [20],
                "mb": [4.1],
                "Qmt": [175],
                "mt": [175],
                "Qref": [91],
                "alphas": [0.118],
                "nfref": [5],
                "HQ": ["MSBAR"],
            }
        )
        self.run(cartesian_product(th), operators.build({"Q2grid": [[100]]}), ["ToyLH"])


class BenchmarkFFNS(ApfelBenchmark):
    """Benckmark FFNS"""

    ffns_theory = {
        "FNS": "FFNS",
        "ModEv": [
            "EXA",
            "EXP",
            "TRN",
        ],
        "NfFF": 4,
        "kcThr": 0.0,
        "kbThr": np.inf,
        "ktThr": np.inf,
        "Q0": 5,
    }
    ffns_theory = tolist(ffns_theory)

    def benchmark_plain(self, pto):
        """Plain configuration"""

        th = self.ffns_theory.copy()
        th.update({"PTO": [pto]})
        self.run(
            cartesian_product(th), operators.build(operators.apfel_config), ["ToyLH"]
        )

    def benchmark_sv(self, pto, svmode):
        """Scale Variation"""

        ts = []
        th = self.ffns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "XIR": [np.sqrt(0.5)],
                "fact_to_ren_scale_ratio": [np.sqrt(2.0)],
                "ModSV": [svmode],
                "EScaleVar": [0],
            }
        )
        ts.extend(cartesian_product(th))
        th = self.ffns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "XIR": [np.sqrt(2.0)],
                "fact_to_ren_scale_ratio": [np.sqrt(0.5)],
                "ModSV": [svmode],
                "EScaleVar": [0],
            }
        )
        ts.extend(cartesian_product(th))
        self.run(ts, operators.build(operators.apfel_config), ["ToyLH"])


class BenchmarkFFNS_qed(ApfelBenchmark):
    """Benckmark FFNS"""

    ffns_theory = {
        "Qref": 91.1870,
        "mc": 1.3,
        "mb": 4.75,
        "mt": 172.0,
        "FNS": "FFNS",
        "ModEv": [
            "EXA",
            # "EXP",
            # "TRN",
        ],
        "NfFF": 5,
        "kcThr": 0.0,
        "kbThr": 0.0,
        "ktThr": np.inf,
        "Q0": 5.0,
        "alphas": 0.118000,
        "alphaqed": 0.007496,
        "alphaem_running": True,
    }
    ffns_theory = tolist(ffns_theory)

    def benchmark_plain(self, pto, qed):
        """Plain configuration"""

        th = self.ffns_theory.copy()
        th.update({"PTO": [pto], "QED": [qed]})
        self.skip_pdfs = lambda _theory: [
            -6,
            6,
            "Tu8",
            "Vu8",
        ]
        self.run(
            cartesian_product(th),
            operators.build(operators.apfel_config),
            ["NNPDF31_nnlo_as_0118_luxqed"],
        )

    def benchmark_sv(self, pto, qed, svmode):
        """Scale Variation"""

        ts = []
        th = self.ffns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "QED": [qed],
                "XIR": [np.sqrt(0.5)],
                "fact_to_ren_scale_ratio": [np.sqrt(2.0)],
                "ModSV": [svmode],
                "EScaleVar": [0],
            }
        )
        ts.extend(cartesian_product(th))
        th = self.ffns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "QED": [qed],
                "XIR": [np.sqrt(2.0)],
                "fact_to_ren_scale_ratio": [np.sqrt(0.5)],
                "ModSV": [svmode],
                "EScaleVar": [0],
            }
        )
        ts.extend(cartesian_product(th))
        self.skip_pdfs = lambda _theory: [
            -6,
            6,
            "Tu8",
            "Vu8",
        ]
        self.run(ts, operators.build(operators.apfel_config), ["ToyLH"])


class BenchmarkVFNS_qed(ApfelBenchmark):
    """Benckmark FFNS"""

    vfns_theory = {
        "Qref": 91.1870,
        "mc": 1.3,
        "mb": 4.75,
        "mt": 172.0,
        "FNS": "VFNS",
        "ModEv": [
            "EXA",
            # "EXP",
            # "TRN",
        ],
        "kcThr": 1.0,
        "kbThr": 1.0,
        "ktThr": 1.0,
        "Q0": 1.25,
        "alphas": 0.118000,
        "alphaqed": 0.007496,
        "alphaem_running": True,
    }
    vfns_theory = tolist(vfns_theory)

    def benchmark_plain(self, pto, qed):
        """Plain configuration"""

        th = self.vfns_theory.copy()
        th.update({"PTO": [pto], "QED": [qed]})
        self.skip_pdfs = lambda _theory: [
            -6,
            6,
            "Tu8",
            "Vu8",
        ]
        self.run(
            cartesian_product(th),
            operators.build(operators.apfel_config),
            ["NNPDF31_nnlo_as_0118_luxqed"],
        )


if __name__ == "__main__":

    # obj = BenchmarkVFNS()
    # obj = BenchmarkFFNS()

    # obj.benchmark_plain(2)
    # obj.benchmark_sv(2, "exponentiated")
    # obj.benchmark_kthr(2)
    # obj.benchmark_msbar(2)

    # obj = BenchmarkFFNS_qed()
    obj = BenchmarkVFNS_qed()
    obj.benchmark_plain(2, 2)
