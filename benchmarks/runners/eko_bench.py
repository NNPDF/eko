# -*- coding: utf-8 -*-
"""
Run multiple eko benchmarks with the same VFNS settings (as LHA)
and and a fine grid
"""
import numpy as np

from ekomark.benchmark.runner import Runner
from eko.interpolation import make_lambert_grid

base_theory = {
    "ModEv": "EXA",
    "Q0": np.sqrt(
        2.0
    ),  # Eq. (30) :cite:`Giele:2002hx`, Eq. (4.53) :cite:`Dittmar:2005ed`
    "nfref": 3,
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
    # TODO: fix this with merging with develop ( not now, will screw db ... )
    # "nf0": 3,
    "kcThr": 1.0 + 1e-15,  # here you need to start in nf=3 and do the match.
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

    def doit(self, pto):
        """
        Enforce operators and PDF

        Parameters
        ----------
            pto : int
                perturbation order
        """
        th = self.theory.copy()
        th.update({"PTO": pto})
        self.run(
            [th],
            [
                {
                    "Q2grid": [1e4],
                    "ev_op_iterations": 100,
                    "interpolation_xgrid": make_lambert_grid(150).tolist(),
                }
            ],
            ["ToyLH"],
        )


if __name__ == "__main__":

    programs = ["LHA", "pegasus", "apfel"]
    for p in programs:
        obj = BenchmarkRunner(p)
        obj.doit(2)
