# -*- coding: utf-8 -*-
"""
    Benchmark EKO to :cite:`Giele:2002hx`
"""
import logging
import sys
import pathlib
from ekomark import lha

here = pathlib.Path(__file__).parents[1]
input_path = here / "input" / "LHA"
theory_card = "LO-EXA.yaml"
operators_card = "iter10-l30m20r4.yaml"

if __name__ == "__main__":
    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko").handlers = []
    logging.getLogger("eko").addHandler(logStdout)
    logging.getLogger("eko").setLevel(logging.INFO)

    app = lha.LHABenchmarkPaper(
        input_path / "theories" / theory_card,
        input_path / "operators" / operators_card,
        here / "assets" / "LHA",
        here / "data",
    )
    app.run()
