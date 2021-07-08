# -*- coding: utf-8 -*-
"""
This script contains a specialization of the Ekomark runner
"""
import functools
import pathlib
import copy
import pandas as pd
import numpy as np

from banana import load_config
from banana.data import dfdict

from ekomark.benchmark.runner import Runner
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

        for column_name, column_data in quark.items():
            if column_name == "x":
                continue
            rot_log[r"${%s}^{+}$" % (pid)][column_name] = (
                column_data + qbar[column_name]
            )
            rot_log[r"${%s}^{-}$" % (pid)][column_name] = (
                column_data - qbar[column_name]
            )
    return rot_log


class EkoRunner(Runner):
    """
    Specialization of the Ekomark runner.
    """

    def __init__(self):
        super().__init__()
        self.banana_cfg = load_config(pkg_path)
        self.rotate_to_evolution_basis = False
        self.plot_pdfs = [4, -4]
        self.sandbox = True
        self.fig_name = None

    def run_external(self, theory, ocard, pdf):

        if self.external == "inputpdf":
            # Compare with the initial pdf

            xgrid = ocard["interpolation_xgrid"]
            q0 = theory["Q0"]
            ext = {}
            pdfs = lhapdf.mkPDFs(pdf.set().name)
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

    def log(self, theory, ocard, _pdf, me, ext):
        """Apply PDFs to eko and produce log tables"""
        log_tabs = {}
        xgrid = ocard["interpolation_xgrid"]
        q2s = ocard["Q2grid"]

        rotate_to_evolution = None
        if self.rotate_to_evolution_basis:
            rotate_to_evolution = eko.basis_rotation.rotate_flavor_to_evolution.copy()

        pdf_name = _pdf.set().name
        pdfs = lhapdf.mkPDFs(pdf_name)

        # build table
        tab = {}
        tab["x"] = xgrid

        # Loop over pdfs replicas
        for rep, pdf in enumerate(pdfs):
            pdf_grid = me.apply_pdf_flavor(
                pdf,
                xgrid,
                flavor_rotation=rotate_to_evolution,
            )

            log_tab = dfdict.DFdict()
            ref_pdfs = ext[rep]

            # Loop over pdf ids
            for key in ref_pdfs:
                if key not in self.plot_pdfs:
                    continue

                if self.external == "inputpdf":
                    tab[f'{pdf_name}_@_{theory["Q0"]}'] = ref_pdfs[key]

                # Loop over q2 grid
                for q2 in q2s:
                    res = pdf_grid[q2]
                    my_pdfs = res["pdfs"]
                    my_pdf_errs = res["errors"]
                    tab[f"EKO_@_{np.round(np.sqrt(q2), 2)}"] = xgrid * my_pdfs[key]
                    tab[f"EKO_error_@_{np.round(np.sqrt(q2), 2)}"] = (
                        xgrid * my_pdf_errs[key]
                    )

                log_tab[pdfname(key)] = pd.DataFrame(tab)
            log_tabs[rep] = log_tab

        # Plot
        new_log = functools.reduce(
            lambda dfd1, dfd2: dfd1.merge(dfd2), log_tabs.values()
        )
        plot_pdf(rotate_to_pm_basis(new_log), self.fig_name, cl=1)
        return new_log
