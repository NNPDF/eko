# -*- coding: utf-8 -*-
"""
    Benchmark EKO to APFEL
"""
import logging
import sys
import pathlib
from ekomark import apfel_benchmark

here = pathlib.Path(__file__).parent

if __name__ == "__main__":
    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko").handlers = []
    logging.getLogger("eko").addHandler(logStdout)
    logging.getLogger("eko").setLevel(logging.INFO)

    # run as cli
    if len(sys.argv) == 3:
        app = apfel_benchmark.ApfelBenchmark(
            sys.argv[1], sys.argv[2], here / "assets" / "APFEL"
        )
        app.run()
    else:
        me = sys.argv[0]
        print(f"Usage: {me} path/to/theory/card.yaml path/to/operators/card.yaml")
