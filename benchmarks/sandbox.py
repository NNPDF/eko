# -*- coding: utf-8 -*-
# pylint: skip-file
import numpy as np

from ekomark import register
from ekomark.benchmark.runner import Runner
from ekomark.data import operators

register(__file__)

vfns = {"FNS": "ZM-VFNS", "mc": 1.51, "mb": 4.92, "mt": 172.5}
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
nnpdf_base_theory = {
    "Qref": 91.2,
    "mc": 1.51,
    "mb": 4.92,
    "mt": 172.5,
    "kcThr": 1.0,
    "kbThr": 1.0,
    "ktThr": 1.0,
    "alphas": 0.118000,
    "FNS": "ZM-VFNS",
    "ModEv": "TRN",
}


class Sandbox(Runner):
    sandbox = True

    # select here the external program between LHA, LHAPDF, apfel, pegasus
    external = "apfel"
    # external = "pegasus"

    # select to plot operators
    plot_operator = True

    rotate_to_evolution_basis = True

    @staticmethod
    def generate_operators():
        ops = {
            "ev_op_iterations": [1],
            # "ev_op_max_order": [20],
            "Q2grid": [[100]],
            # "debug_skip_singlet": [True],
        }
        return ops

    def doit(self):
        theory_updates = {
            **ffns3,
            "PTO": 0,
            # "ModEv": "EXA",
            # "XIR": 0.5,
            # "fact_to_ren_scale_ratio": 2.0,
            "Q0": 1.65,  # np.sqrt(10),
            # "Qref": 1.5,
            # "alphas": 0.35,
            # "kbThr": 2.71,
        }
        # t0 = theory_updates.copy()
        # t0["PTO"] = 0
        #    self.skip_pdfs = lambda _theory: [
        #        22,
        #        -6,
        #        6,
        #        "ph",
        #        "V35",
        #        "V24",
        #        "V15",
        #        "V8",
        #        "T35",
        #    ]
        self.skip_pdfs = lambda _theory: [
            22,
            -6,
            6,
            108,
            115,
            224,
            235,
            124,
            135,
            208,
            215,
        ]

        self.run(
            [theory_updates],  # , t0],
            operators.build(self.generate_operators()),
            [
                #    "ToyLH",
                "NNPDF40_nnlo_as_01180",
            ],
        )

    def lha(self):
        theory_updates = {
            "PTO": 0,
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
