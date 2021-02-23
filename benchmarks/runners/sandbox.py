# -*- coding: utf-8 -*-
from banana.data import power_set

from ekomark.benchmark.runner import Runner
from ekomark.data import operators


class Sandbox(Runner):

    """
    Globally set the external program
    """
    sandbox = True

    # select here the external program
    external = "LHA"
    external = "LHAPDF"
    # external = "apfel"

    # select to plot operators
    plot_operator = True

    rotate_to_evolution_basis = True

    # pdf to skip, for LHA there is a default
    skip_pdfs = [22, -6, 6, -5, 5, -4, 4, "ph", "V35", "V24", "V15", "V8", "T35"]
    if external == "LHA":
        skip_pdfs = [22, -6, 6, "ph", "V35", "V24", "V15", "V8", "T35"]

    @staticmethod
    def generate_operators():

        ops = {
            "ev_op_iterations": [10],
            "Q2grid": [[100]],
            "ev_op_max_order": [10],
        }
        return ops

    def run_sand(self):

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
    sand.run_sand()
