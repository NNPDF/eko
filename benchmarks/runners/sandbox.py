# -*- coding: utf-8 -*-
"""
Benchmark EKO to APFEL
"""
import logging
import sys
import pathlib
from ekomark import apfel_benchmark

from ekomark.data.operators import OperatorsGenerator

here = pathlib.Path(__file__).parents[1]
input_path = here / "input" / "APFEL"
theory_card = "NLO-EXA.yaml"
operators_card = "1e4-iter10-l30m20r4.yaml"

def generate_operators():
    og = OperatorsGenerator("sandbox")
    defaults = dict(
        interpolation_is_log=True,
        interpolation_polynomial_degree=4,
        interpolation_xgrid=["make_grid", 30, 20],
        ev_op_max_order=10,
        ev_op_iterations=10,
    )
    defaults["Q2grid"] = 10
    cards = [defaults]
    og.write(cards)

class Sandbox():
    def run_lo(self):
        pass

    def run_nlo(self):
        pass


if __name__ == "__main__":
    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko").handlers = []
    logging.getLogger("eko").addHandler(logStdout)
    logging.getLogger("eko").setLevel(logging.INFO)

    generate_operators()
    # app = apfel_benchmark.ApfelBenchmark(
    #     input_path / "theories" / theory_card,
    #     input_path / "operators" / operators_card,
    #     here / "assets" / "sandbox",
    # )
    # app.run()
