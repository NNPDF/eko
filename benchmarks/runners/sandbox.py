# -*- coding: utf-8 -*-
import numpy as np

from ekomark.benchmark.runner import Runner
from ekomark.data import operators

pineappl_xgrid = list(
    reversed(
        [
            1,
            0.9309440808717544,
            0.8627839323906108,
            0.7956242522922756,
            0.7295868442414312,
            0.6648139482473823,
            0.601472197967335,
            0.5397572337880445,
            0.4798989029610255,
            0.4221667753589648,
            0.3668753186482242,
            0.31438740076927585,
            0.2651137041582823,
            0.2195041265003886,
            0.17802566042569432,
            0.14112080644440345,
            0.10914375746330703,
            0.08228122126204893,
            0.060480028754447364,
            0.04341491741702269,
            0.030521584007828916,
            0.02108918668378717,
            0.014375068581090129,
            0.009699159574043398,
            0.006496206194633799,
            0.004328500638820811,
            0.0028738675812817515,
            0.0019034634022867384,
            0.0012586797144272762,
            0.0008314068836488144,
            0.0005487795323670796,
            0.00036205449638139736,
            0.00023878782918561914,
            0.00015745605600841445,
            0.00010381172986576898,
            6.843744918967896e-05,
            4.511438394964044e-05,
            2.97384953722449e-05,
            1.9602505002391748e-05,
            1.292101569074731e-05,
            8.516806677573355e-06,
            5.613757716930151e-06,
            3.7002272069854957e-06,
            2.438943292891682e-06,
            1.607585498470808e-06,
            1.0596094959101024e-06,
            6.984208530700364e-07,
            4.6035014748963906e-07,
            3.034304765867952e-07,
            1.9999999999999954e-07,
        ]
    )
)


class Sandbox(Runner):

    """
    Globally set the external program
    """

    sandbox = True

    # select here the external program between LHA, LHAPDF, apfel
    external = "apfel"
    # external = "LHA"

    # select to plot operators
    # plot_operator = True

    rotate_to_evolution_basis = True

    # pdf to skip, for LHA there is a default
    def skip_pdfs(self, _theory):
        if self.external == "LHA":
            return [
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
        return [
            22,
            -6,
            6,
            -5,
            5,
            "ph",
            "V35",
            "V24",
            "V15",
            "T35",
            "T24",
        ]

    @staticmethod
    def generate_operators():

        ops = {
            "ev_op_iterations": [10],
            "ev_op_max_order": [10],
            "Q2grid": [[10000]],
        }
        return ops

    def doit(self):

        theory_updates = {
            "PTO": 2,
            "FNS": "FFNS",
            "NfFF": 4,
            "ModEv": "EXA",
            "XIR": 1.4142135623730951,
            "Q0": np.sqrt(2),
            "kcThr": 0.0,
            "kbThr": np.inf,
            "ktThr": np.inf,
            "Qref": np.sqrt(2.0),
            "alphas": 0.35,
        }
        self.run(
            [theory_updates],
            operators.build(self.generate_operators()),
            ["ToyLH"],
        )

    def lha(self):
        theory_updates = {
            "PTO": 1,
            "FNS": "FFNS",
            "NfFF": 4,
            "ModEv": "EXA",
            "XIR": np.sqrt(2),
            "Q0": np.sqrt(2),
            "kcThr": 0.0,
            "kbThr": np.inf,
            "ktThr": np.inf,
            "Qref": np.sqrt(2.0),
            "alphas": 0.35,
        }
        #  theory_updates = {
        #  "PTO": 0,
        #  "FNS": "ZM-VFNS",
        #  "ModEv": "EXA",
        #  # "XIR": np.sqrt(2),
        #  "Q0": np.sqrt(2),
        #  "kcThr": 1.0,
        #  "kbThr": 1.0,
        #  "ktThr": 1.0,
        #  "mc": np.sqrt(2.0),  # Eq. (34)
        #  "mb": 4.5,
        #  "mt": 175,
        #  "Qref": np.sqrt(2.0),
        #  "alphas": 0.35,
        #  }
        self.run(
            [theory_updates],
            [{"Q2grid": [1e4], "debug_skip_singlet": True}],
            ["ToyLH"],
        )


if __name__ == "__main__":
    sand = Sandbox()
    sand.lha()
