"""
Benchmark to Pegasus :cite:`Vogt:2004ns`
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


class PegasusBenchmark(Runner):
    """
    Globally set the external program to Pegasus.

    `imodev = 1` exactly corresponds to `perturbative-exact`.
    However, the difference between `perturbative-exact` and `iterate-exact` is
    subleading (~ 1e-4 relative difference).
    """

    external = "pegasus"

    # Rotate to evolution basis
    rotate_to_evolution_basis = True

    @staticmethod
    def skip_pdfs(_theory):
        # pdf to skip
        return [22, "ph", "T35", "V35"]


class BenchmarkVFNS(PegasusBenchmark):
    """Benckmark VFNS"""

    zm_theory = {
        "FNS": "ZM-VFNS",
        "ModEv": [
            # "perturbative-exact",
            "EXA"
            # "EXP",
            # "ordered-truncated",
        ],
        "kcThr": 1.0,
        "kbThr": 1.0,
        "ktThr": 1.0,
        "Qref": np.sqrt(2.0),
        "Qedref": 0.0,
        "alphas": 0.35,
        "alphaqed": 0.007496,
        "QED": 0,
        "Q0": np.sqrt(2.0),
        "nfref": 3,
        "nf0": 3,
    }
    vfns_theory = tolist(zm_theory)

    def benchmark_plain(self, pto):
        """Plain configuration"""

        th = self.vfns_theory.copy()
        th.update(
            {
                "PTO": [pto],
            }
        )
        self.run(
            cartesian_product(th), operators.build(operators.pegasus_config), ["ToyLH"]
        )

    def benchmark_sv(self, pto, svmode):
        """Scale Variation"""

        th = self.vfns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "XIF": [np.sqrt(0.5), np.sqrt(2.0)],
                "ModSV": [svmode],
            }
        )
        self.run(
            cartesian_product(th), operators.build(operators.pegasus_config), ["ToyLH"]
        )


class BenchmarkFFNS(PegasusBenchmark):
    """Benckmark FFNS"""

    ffns_theory = {
        "FNS": "FFNS",
        "ModEv": [
            "perturbative-exact",
            # "EXP",
            # "ordered-truncated",
        ],
        "NfFF": 3,
        "kcThr": np.inf,
        "kbThr": np.inf,
        "ktThr": np.inf,
        "Qref": np.sqrt(2.0),
        "Qedref": 0.0,
        "alphas": 0.35,
        "alphaqed": 0.007496,
        "Q0": np.sqrt(2.0),
    }
    ffns_theory = tolist(ffns_theory)

    def benchmark_plain(self, pto):
        """Plain configuration"""

        th = self.ffns_theory.copy()
        th.update(
            {
                "PTO": [pto],
            }
        )
        self.run(
            cartesian_product(th), operators.build(operators.pegasus_config), ["ToyLH"]
        )

    def benchmark_plain_pol(self, pto):
        """Plain polarized configuration"""

        th = self.ffns_theory.copy()
        th.update({"PTO": [pto]})
        op = operators.pegasus_config.copy()
        op["polarized"] = [True]
        self.run(cartesian_product(th), operators.build(op), ["ToyLH_polarized"])

    def benchmark_sv(self, pto, svmode):
        """Scale Variation"""

        th = self.ffns_theory.copy()
        th.update(
            {
                "PTO": [pto],
                "XIF": [np.sqrt(0.5), np.sqrt(2.0)],
                "ModSV": [svmode],
            }
        )
        self.run(
            cartesian_product(th), operators.build(operators.pegasus_config), ["ToyLH"]
        )


if __name__ == "__main__":
    obj = BenchmarkVFNS()
    # obj = BenchmarkFFNS()
    # obj.benchmark_plain_pol(1)
    # obj.benchmark_plain(1)

    obj.benchmark_sv(1, "exponentiated")
    # vfns.benchmark_sv()
