# -*- coding: utf-8 -*-
import pathlib

from banana.data import power_set

from ekomark.benchmark.runner import Runner
from ekomark.data import operators


class Sandbox(Runner):

    """
    Globally set the external program 
    """

    external = "LHA"
    external = "LHAPDF"
    #external = "apfel"

    # selcet output type:
    post_process_config = {
        "plot_PDF": False,
        # TODO: fix this still not working!!! there are no labels in the output!!
        "plot_operator": False,
        "write_operator": True,
    }

    # TODO: better move this to the runner ?
    # output dir
    output_path = (
        f"{pathlib.Path(__file__).absolute().parents[1]}/data/{external}_bench"
    )

    # Rotate to evolution basis
    rtevb = True

    @staticmethod
    def generate_operators():

        ops = {
            "ev_op_iterations": [2],
            "Q2grid": [[2,10]],
            "ev_op_max_order": [2],
        }
        return ops

    def _run(self):

        theory_updates = {
            "PTO": [0],
            "FNS": ["FFNS",],
            "NfFF": [3,],
            "ModEv": ["EXA",],
            "XIR": [1.4142135623730951,],
            "alphas": [0.35],
            "Qref": [1.4142135623730951],
        }
        self.run(
            power_set(theory_updates),
            operators.build(self.generate_operators()),
            ["ToyLH"],
        )


if __name__ == "__main__":

    sand = Sandbox()
    sand._run()
