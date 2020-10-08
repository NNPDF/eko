# -*- coding: utf-8 -*-
"""
    Benchmark EKO to APFEL
"""
import logging
import sys

import ekomark.apfel_benchmark

if __name__ == "__main__" and True:
    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko").handlers = []
    logging.getLogger("eko").addHandler(logStdout)
    logging.getLogger("eko").setLevel(logging.INFO)

    # run as cli
    if len(sys.argv) == 2:
        app = ekomark.apfel_benchmark.ApfelBenchmark(sys.argv[1])
        app.run()
    else:
        me = sys.argv[0]
        print(f"Usage: {me} path/to/input/card.yaml")
