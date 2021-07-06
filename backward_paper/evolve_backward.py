# -*- coding: utf-8 -*-
"""
This script compute an EKO to evolve a PDF set under the charm thrshold replica by replica
"""
import functools
import logging
import pathlib
import os
import sys
import copy
import pandas as pd
import numpy as np

from banana import load_config
from banana.benchmark.runner import BenchmarkRunner, get_pdf
from banana.data import dfdict

from ekomark.data import operators, db
from ekomark import pdfname

from plots import plot_pdf

import lhapdf
import eko

pkg_path = pathlib.Path(__file__).absolute().parents[0]


def rotate_to_pm_basis(log):
    """Rotate to plus minus basis"""
    # TODO: this can be improved
    rot_log = {}
    if "g" in log:
        rot_log["g"] = log["g"]
    for pid in eko.evolution_operator.flavors.quark_names:
        if pid not in log:
            continue
        quark = log[pid]
        qbar = log[f"{pid}bar"].copy()
        rot_log[r"${%s}^{+}$" % (pid)] = copy.deepcopy(log[pid])
        rot_log[r"${%s}^{-}$" % (pid)] = copy.deepcopy(log[pid])
        rot_log[r"${%s}^{+}$" % (pid)].eko = (quark + qbar).eko
        rot_log[r"${%s}^{-}$" % (pid)].eko = (quark - qbar).eko
        rot_log[r"${%s}^{+}$" % (pid)].inputpdf = (quark + qbar).inputpdf
        rot_log[r"${%s}^{-}$" % (pid)].inputpdf = (quark - qbar).inputpdf
    return rot_log


class Runner(BenchmarkRunner):
    """
    EKO specialization of the banana runner.

    Prameters
    ---------
        pdf_name: str
            PDF name
    """

    def __init__(self, pdf_name):
        super().__init__()
        self.banana_cfg = load_config(pkg_path)
        self.db_base_cls = db.Base
        self.rotate_to_evolution_basis = False
        self.pdf_name = pdf_name
        self.skip_pdfs = [21, -1, 1, -2, 2, -3, 3, 5, -5, -6, 6, 22]

    @staticmethod
    def load_ocards(session, ocard_updates):
        return operators.load(session, ocard_updates)

    def run_external(self, theory, ocard, _pdf):
        """Store the initial pdf and its uncertanty"""
        xgrid = ocard["interpolation_xgrid"]
        q0 = theory["Q0"]
        ext = {}

        # is the set installed? if not do it now
        get_pdf(self.pdf_name)
        pdfs = lhapdf.mkPDFs(self.pdf_name)
        for rep, base_pdf in enumerate(pdfs):
            tab = {}
            for x in xgrid:
                in_pdf = base_pdf.xfxQ(x, q0)
                for pid, val in in_pdf.items():
                    if pid not in tab:
                        tab[pid] = []
                    tab[pid].append(val)

            ext[rep] = tab
        return ext

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

    def log(self, theory, ocard, _pdf, me, ext):
        """Apply PDFs to eko and produce log tables"""
        log_tabs = {}
        xgrid = ocard["interpolation_xgrid"]
        # assume to have just one q2 in the grid
        q2 = ocard["Q2grid"][0]

        rotate_to_evolution = None
        if self.rotate_to_evolution_basis:
            rotate_to_evolution = eko.basis_rotation.rotate_flavor_to_evolution.copy()

        pdfs = lhapdf.mkPDFs(self.pdf_name)

        # Loop over pdfs replicas
        for rep, pdf in enumerate(pdfs):
            pdf_grid = me.apply_pdf_flavor(
                pdf,
                xgrid,
                flavor_rotation=rotate_to_evolution,
            )

            log_tab = dfdict.DFdict()
            ref_pdfs = ext[rep]
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
                tab["inputpdf"] = ref_pdfs[key]
                # tab["percent_error"] = (f - r) / r * 100

                log_tab[pdfname(key)] = pd.DataFrame(tab)
            log_tabs[rep] = log_tab

        # Plot
        new_log = functools.reduce(
            lambda dfd1, dfd2: dfd1.merge(dfd2), log_tabs.values()
        )
        plot_pdf(rotate_to_pm_basis(new_log), self.pdf_name, theory["Q0"], cl=1)
        return new_log


if __name__ == "__main__":

    theory_updates = {
        "Qref": 9.1187600e01,
        "alphas": 0.1180024,
        "mc": 1.51,
        "mb": 4.92,
        "mt": 172.5,
        "kcThr": 1.0,
        "kbThr": 1.0,
        "ktThr": 1.0,
        "PTO": 2,
        "Q0": 1.65,
        "IC": 1,
    }
    operator_updates = {
        "interpolation_xgrid": np.linspace(0.01, 1, 100),
        "Q2grid": [1.50 ** 2],
        "backward_inversion": "expanded",
    }

    pdf_names = [
        "210629-n3fit-001",  # NNLO, fitted charm
        "210629-theory-003",  # NNLO, perturbative charm
        "210701-n3fit-data-014",  # NNLO, fitted charm + EMC F2c
    ]

    for name in pdf_names:
        backward_runner = Runner(name)
        backward_runner.run([theory_updates], [operator_updates], [name])
