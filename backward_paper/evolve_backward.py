# -*- coding: utf-8 -*-
"""
This script compute an EKO to evolve a PDF set under the charm thrshold replica by replica
"""
import functools
import logging
import pathlib
import os
import sys
import pandas as pd
import numpy as np

from banana import load_config
from banana.benchmark.runner import BenchmarkRunner
from banana.data import dfdict

from ekomark.data import operators, db
from ekomark import pdfname
from eko import basis_rotation as br

import eko
import lhapdf

from matplotlib import use, rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

pkg_path = pathlib.Path(__file__).absolute().parents[0]


def plot_pdf(log, pdf_name):

    path = pkg_path / f"{pdf_name}.pdf"
    print(f"Writing pdf plots to {path}")

    with PdfPages(path) as pp:
        for name, vals in log.items():

            fig = plt.figure(figsize=(15, 5))
            plt.title(name)

            mean = vals.groupby(['x'], axis=0, as_index=False).mean()
            std = vals.groupby(['x'], axis=0, as_index=False).std()
            mean.plot('x', 'eko')
            plt.fill_between(std.x, mean.eko - std.eko, mean.eko + std.eko, alpha=0.2)

            plt.xscale("log")
            plt.ylabel(r"$\rm{ x%s(x)}$"%name, fontsize=11)
            plt.xlabel("x")
            plt.plot(np.geomspace(1e-7,1,200), np.zeros(200), "k--", alpha=0.7)
            #plt.legend(f"{pdf_name} @ 1.5 GeV")
            plt.tight_layout()

            pp.savefig()
            plt.close(fig)

class Runner(BenchmarkRunner):
    """
    EKO specialization of the banana runner.
    """

    banana_cfg = load_config(pkg_path)
    db_base_cls = db.Base
    rotate_to_evolution_basis = False
    pdf_name="NNPDF31_nnlo_as_0118"
    skip_pdfs=[5,-5,-6,6,22]

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

    def log(self, _theory, ocard, _pdf, me, _ext):

        log_tabs = {}
        xgrid = ocard["interpolation_xgrid"]
        # assume to have just one q2 in the grid
        q2 = ocard["Q2grid"][0]

        rotate_to_evolution = None
        if self.rotate_to_evolution_basis:
            rotate_to_evolution = br.rotate_flavor_to_evolution.copy()

        pdfs = lhapdf.mkPDFs(self.pdf_name)

        # Loop over pdfs replicas
        for rep, pdf in enumerate(pdfs):
            pdf_grid = me.apply_pdf_flavor(
                pdf,
                xgrid,
                flavor_rotation=rotate_to_evolution,
            )

            log_tab = dfdict.DFdict()
            #ref_pdfs = ext["values"][q2]
            res = pdf_grid[q2]
            my_pdfs = res["pdfs"]
            my_pdf_errs = res["errors"]

            for key in my_pdfs:
                if key in self.skip_pdfs:
                    continue
                # build table
                tab = {}
                tab["x"] = xgrid
                tab["Q2"] = q2
                tab["eko"] = xgrid * my_pdfs[key]
                tab["eko_error"] = xgrid * my_pdf_errs[key]
                #tab[self.external] = r = ref_pdfs[key]
                #tab["percent_error"] = (f - r) / r * 100

                log_tab[pdfname(key)] = pd.DataFrame(tab)
            log_tabs[rep] = log_tab

        # Plot
        new_log = functools.reduce(lambda dfd1, dfd2: dfd1.merge(dfd2), log_tabs.values())
        plot_pdf(new_log, self.pdf_name)
        return new_log


if __name__ == "__main__":

    backward_runner = Runner()
    theory_updates = {
        "Qref": 91.2,
        "alphas": 0.118000,
        "mc": 1.51,
        "mb": 4.92,
        "mt": 172.5,
        "kcThr": 1.0,
        "kbThr": 1.0,
        "ktThr": 1.0,
        "PTO": 2,
        "Q0": 1.65,
    }
    operator_updates = {
        "Q2grid": [1.5],
        "backward_inversion": "exact",
    }
    
    # toyLH is not used in log
    backward_runner.run([theory_updates], [operator_updates], ["ToyLH"])
