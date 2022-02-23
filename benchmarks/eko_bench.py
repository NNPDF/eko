# -*- coding: utf-8 -*-
"""
Run multiple eko benchmarks with the same VFNS settings (as LHA)
and and a fine grid
"""
import numpy as np
from banana import register

from eko.interpolation import make_lambert_grid
from ekomark.benchmark.runner import Runner

register(__file__)

base_theory = {
    "ModEv": "EXA",
    "Q0": np.sqrt(
        2.0
    ),  # Eq. (30) :cite:`Giele:2002hx`, Eq. (4.53) :cite:`Dittmar:2005ed`
    "nfref": 3,
    "nf0": 3,
    "mc": np.sqrt(
        2.0
    ),  # Eq. (34) :cite:`Giele:2002hx`, Eq. (4.56) :cite:`Dittmar:2005ed`
    "mb": 4.5,
    "mt": 175.0,
    "Qref": np.sqrt(
        2.0
    ),  # Eq. (32) :cite:`Giele:2002hx`,Eq. (4.53) :cite:`Dittmar:2005ed`
    "alphas": 0.35,  # Eq. (4.55) :cite:`Dittmar:2005ed`
    "FNS": "ZM-VFNS",  # ignored by eko, but needed by LHA_utils
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
}
"""Global theory settings"""


class BenchmarkRunner(Runner):
    """
    Generic benchmark runner using the LHA VFNS settings
    """

    def __init__(self, external):
        super().__init__()
        self.external = external
        self.theory = base_theory
        self.sandbox = True
        self.rotate_to_evolution_basis = True

    def skip_pdfs(self, _theory):
        return [22, -6, 6, "ph", "V35", "V24", "V15", "V8", "T35"]

    def run_lha(self, theory_updates):
        """
        Enforce operators and PDF

        Parameters
        ----------
            theory_updates : list(dict)
                theory updates
        """
        self.run(
            theory_updates,
            [
                {
                    "Q2grid": [1e4],
                    "ev_op_iterations": 10,
                    "interpolation_xgrid": make_lambert_grid(60).tolist(),
                }
            ],
            ["ToyLH"],
        )

    def benchmark_sv(self, pto):
        """
        Scale variations

        Parameters
        ----------
            pto : int
                perturbation order

        Returns
        -------
            list(dict)
                theory updates
        """
        low = base_theory.copy()
        low["PTO"] = pto
        low["fact_to_ren_scale_ratio"] = np.sqrt(1.0 / 2.0)
        low["ModSV"] = "exponentiated"
        # needed for apfel
        low["XIR"] = np.sqrt(2.0)
        low["EScaleVar"] = 0
        high = base_theory.copy()
        high["PTO"] = pto
        high["fact_to_ren_scale_ratio"] = np.sqrt(2.0)
        high["ModSV"] = "A"
        # needed for apfel
        high["XIR"] = np.sqrt(0.5)
        high["EScaleVar"] = 0
        self.run_lha([low, high])

    def benchmark_plain(self, pto):
        """Plain configuration"""
        th = self.theory.copy()
        th.update({"PTO": pto})
        self.run_lha([th])


if __name__ == "__main__":

    programs = ["LHA", "pegasus", "apfel"]
    for p in programs:
        obj = BenchmarkRunner(p)
        # obj.benchmark_plain(2)
        obj.benchmark_sv(2)
