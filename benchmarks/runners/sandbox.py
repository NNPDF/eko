# -*- coding: utf-8 -*-
# pylint: skip-file
import numpy as np

from ekomark.benchmark.runner import Runner
from ekomark.data import operators

vfns = {"FNS": "ZM-VFNS", "mc": 1e4, "mb": 1.5e4, "mt": 2e4}
pegasus_vfns = {"nfref": 3, **vfns}
ffns3 = {
    "kcThr": np.inf,
    "kbThr": np.inf,
    "ktThr": np.inf,
    "NfFF": 3,
    "FNS": "FFNS",
}
ffns4 = {
    "kcThr": 0.0,
    "kbThr": np.inf,
    "ktThr": np.inf,
    "NfFF": 4,
    "FNS": "FFNS",
}


class Sandbox(Runner):

    """
    Globally set the external program
    """

    sandbox = True

    # select here the external program between LHA, LHAPDF, apfel, pegasus
    # external = "apfel"
    external = "pegasus"

    # select to plot operators
    plot_operator = False

    rotate_to_evolution_basis = True

    @staticmethod
    def generate_operators():
        ops = {
            "ev_op_iterations": [15],
            "ev_op_max_order": [20],
            "Q2grid": [[1e3]],
            # "debug_skip_singlet": [True],
        }
        return ops

    def doit(self):
        theory_updates = {
            "PTO": 2,
            "ModEv": "perturbative-exact",
            # "XIR": 0.5,
            # "fact_to_ren_scale_ratio": 2.0,
            "Q0": np.sqrt(2),
            "Qref": np.sqrt(2.0),
            "alphas": 0.35,
            **ffns3,
        }
        # t0 = theory_updates.copy()
        # t0["PTO"] = 0
        self.skip_pdfs = lambda _theory: [
            22,
            -6,
            6,
            "ph",
            "V35",
            "V24",
            "V15",
            "V8",
            "T35",
        ]
        self.run(
            [theory_updates],  # , t0],
            operators.build(self.generate_operators()),
            ["ToyLH"],
        )

    def lha(self):
        theory_updates = {
            "PTO": 1,
            "FNS": "FFNS",
            "NfFF": 4,
            "ModEv": "EXA",
            "fact_to_ren_scale_ratio": np.sqrt(2),
            "Q0": np.sqrt(2),
            "kcThr": 0.0,
            "kbThr": np.inf,
            "ktThr": np.inf,
            "Qref": np.sqrt(2.0),
            "alphas": 0.35,
        }
        self.skip_pdfs = lambda _theory: [
            22,
            -6,
            6,
            "ph",
            "V35",
            "V24",
            "V15",
            "V8",
            "T35",
        ]
        self.run(
            [theory_updates],
            [{"Q2grid": [1e4], "debug_skip_singlet": True}],
            ["ToyLH"],
        )


if __name__ == "__main__":
    sand = Sandbox()
    sand.doit()
