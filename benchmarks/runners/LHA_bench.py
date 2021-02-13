# -*- coding: utf-8 -*-
"""
    Benchmark EKO to :cite:`Giele:2002hx`
"""
import logging
import sys
import pathlib


from ekomark.benchmark.runner import Runner
from ekomark.data import operators

class LHABenchmark(Runner):

    """
    Globally set the external program to LHA
    """

    external = "LHA"
    
    #TODO: place this somewhere
    post_process_config = {
        "plot_PDF": False,
        "plot_operator": False,  # True,
        "write_operator": False,
    }

    assets_dir = str(pathlib.Path(__file__).parents[0])
    output_path = str(pathlib.Path(__file__).parents[0])
    ref = {}

class BenchmarkPlain(LHABenchmark):
    """The most basic checks"""

    def benchmark_lo(self):
        self.run([{}], [operators.default_config[0]], ["ToyLH"])

    def benchmark_nlo(self):
        self.run([{'PTO': 1}], [operators.default_config[0]], ["ToyLH"])


if __name__ == "__main__":

    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko").handlers = []
    logging.getLogger("eko").addHandler(logStdout)
    logging.getLogger("eko").setLevel(logging.INFO)

    lha = BenchmarkPlain()
    lha.benchmark_lo()
    #lha.benchmark_nlo()
    
