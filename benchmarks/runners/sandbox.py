# -*- coding: utf-8 -*-
"""
Benchmark EKO to APFEL
"""
import logging
import sys
import pathlib
from ekomark import apfel_benchmark

here = pathlib.Path(__file__).parents[1]
input_path = here / "input" / "APFEL"
theory_card = "NLO-EXA.yaml"
operators_card = "1e4-iter10-l30m20r4.yaml"

if __name__ == "__main__":
    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko").handlers = []
    logging.getLogger("eko").addHandler(logStdout)
    logging.getLogger("eko").setLevel(logging.INFO)

    app = apfel_benchmark.ApfelBenchmark(
        input_path / "theories" / theory_card,
        input_path / "operators" / operators_card,
        here / "assets" / "APFEL",
    )
    app.run()
