# -*- coding: utf-8 -*-
"""
This script compute an EKO to evolve a PDF set under the charm thrshold replica by replica
"""
import logging
import pathlib
import os
import sys

from banana import load_config
from banana.benchmark.runner import BenchmarkRunner

from ekomark.data import operators, db
import eko

pkg_path = pathlib.Path(__file__).absolute().parents[0]


class Runner(BenchmarkRunner):
    """
    EKO specialization of the banana runner.
    """

    banana_cfg = load_config(pkg_path)
    db_base_cls = db.Base
    rotate_to_evolution_basis = False

    @staticmethod
    def load_ocards(session, ocard_updates):
        return operators.load(session, ocard_updates)

    def run_external(self, theory, ocard, pdf):
        pass

    def run_me(self, theory, ocard, _pdf):
        """
        Run eko

        Parameters
        ----------
            theory : dict
                theory card
            ocard : dict
                operator card

        Returns
        -------
            out :  dict
                DGLAP result
        """

        # activate logging
        logStdout = logging.StreamHandler(sys.stdout)
        logStdout.setLevel(logging.INFO)
        logStdout.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger("eko").handlers = []
        logging.getLogger("eko").addHandler(logStdout)
        logging.getLogger("eko").setLevel(logging.INFO)

        ops_id = f"o{ocard['hash'][:6]}_t{theory['hash'][:6]}"
        path = f"{self.banana_cfg['database_path'].parents[0]}/{ops_id}.yaml"
        rerun = True
        if os.path.exists(path):
            rerun = False
            ask = input("Use cached output? [Y/n]")
            if ask.lower() in ["n", "no"]:
                rerun = True

        if rerun:
            out = eko.run_dglap(theory, ocard)
            print(f"Writing operator to {path}")
            out.dump_yaml_to_file(path)
        else:
            print(f"Using cached eko data: {os.path.relpath(path,os.getcwd())}")
            with open(path) as o:
                out = eko.output.Output.load_yaml(o)

        return out

    def log(self, theory, ocard, pdf, me, ext):
        # TODO: combine here the eko with pdf replicas
        # and then plot
        pass


if __name__ == "__main__":

    backward_runner = Runner()
    # TODO: add here the correct parameters
    theory_updates = {
        "Qref": 91.2,
        "alphas": 0.118000,
        "mc": 1.51,
        "mb": 4.92,
        "mt": 172.5,
        "kcThr": 1.0,
        "kbThr": 1.0,
        "ktThr": 1.0,
        "PTO": 1,
        "Q0": 1.7,
    }
    operator_updates = {
        "Q2grid": [1.5],
        "backward_inversion": "exact",
    }

    backward_runner.run([theory_updates], [operator_updates], ["ToyLH"])
