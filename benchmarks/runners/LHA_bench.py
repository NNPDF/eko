# -*- coding: utf-8 -*-
"""
    Benchmark EKO to :cite:`Giele:2002hx`
"""
import logging
import sys
import pathlib


# from ekomark import LHA

from ekomark.benchmark.runner import Runner
from ekomark.data import observables

here = pathlib.Path(__file__).parent

class LHABenchmark(Runner):

    """
    Globally set the external program to LHA
    """

    external = "LHA"
    post_process_config = {
        "plot_PDF": True,
        "plot_operator": False,  # True,
        "write_operator": True,
    }


class BenchmarkPlain(LHABenchmark):
    """The most basic checks"""

    def benchmark_lo(self):
        self.run([{}], observables.build(**(observables.default_config[0])))

    def benchmark_nlo(self):
        self.run([{'PTO': 1}], observables.build(**(observables.default_config[0])))


if __name__ == "__main__":
    
    lha = BenchmarkPlain()
    lha.benchmark_lo()
    lha.benchmark_nlo()
    
    
    # activate logging
    #logStdout = logging.StreamHandler(sys.stdout)
    #logStdout.setLevel(logging.INFO)
    #logStdout.setFormatter(logging.Formatter("%(message)s"))
    #logging.getLogger("eko").handlers = []
    #logging.getLogger("eko").addHandler(logStdout)
    #logging.getLogger("eko").setLevel(logging.INFO)

    # run as cli
    #if len(sys.argv) == 3:
    #    app = LHA.LHABenchmarkPaper(
    #        sys.argv[1],
    #        sys.argv[2],
    #        here / "assets" / "LHA",
    #        here / "data",
    #    )
    #    app.run()
    #else:
    #    me = sys.argv[0]
    #    print(f"Usage: {me} path/to/theory/card.yaml path/to/operators/card.yaml")
