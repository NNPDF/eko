# -*- coding: utf-8 -*-
"""
    Benchmark EKO to :cite:`Giele:2002hx`
"""
import logging
import sys
import pathlib
from ekomark import lhapdf_benchmark

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
    if len(sys.argv) == 4:
        app = lhapdf_benchmark.LHAPDFBenchmark(
            sys.argv[1],
            sys.argv[2],
            sys.argv[3],
            here / "assets" / "LHAPDF",
        )
        app.run()
    else:
        me = sys.argv[0]
        print(f"Usage: {me} path/to/theory/card.yaml path/to/operators/card.yaml pdf")
