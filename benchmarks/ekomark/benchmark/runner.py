# -*- coding: utf-8 -*-
"""
Abstract layer for running the benchmarks
"""
import pprint
import io
import os
import logging
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from banana.data import sql, dfdict
from banana.benchmark.runner import BenchmarkRunner

from ekomark.banana_cfg import banana_cfg
from ekomark.data import operators
from ekomark.plots import plot_dist, plot_operator

import eko


def pdfname(pid_or_name):
    """ Return pdf name  """
    if isinstance(pid_or_name, int):
        return eko.basis_rotation.flavor_basis_names[
            eko.basis_rotation.flavor_basis_pids.index(pid_or_name)
        ]
    return pid_or_name


class Runner(BenchmarkRunner):
    banana_cfg = banana_cfg
    rotate_to_evolution_basis = False
    skip_pdfs = []

    @staticmethod
    def init_ocards(conn):
        with conn:
            conn.execute(sql.create_table("operators", operators.default_card))

    @staticmethod
    def load_ocards(conn, ocard_updates):
        return operators.load(conn, ocard_updates)

    def run_me(self, theory, ocard, pdf):
        """
        Run eko

        Parameters
        ----------
            theory : dict
                theory card
            ocard : dict
                operator card
            pdf : lhapdf_like
                PDF set

        Returns
        -------
            out :  dict
                DGLAP result
        """

        # TODO: check cache using banana ? if sandbox check for cache.
        # target_path = self.assets_dir / (self.output_path + "-ops.yaml")
        # rerun = True
        # if target_path.exists():
        #    rerun = False
        #    ask = input("Use cached output? [Y/n]")
        #    if ask.lower() in ["n", "no"]:
        #        rerun = True
        # if rerun:
        #    ret = eko.run_dglap(self.theory, self.operators)
        # else:
        #    # load
        #    self.post_process_config["write_operator"] = False
        #    with open(target_path) as o:
        #        ret = eko.output.Output.load_yaml(o)

        # activate logging
        logStdout = logging.StreamHandler(sys.stdout)
        logStdout.setLevel(logging.INFO)
        logStdout.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger("eko").handlers = []
        logging.getLogger("eko").addHandler(logStdout)
        logging.getLogger("eko").setLevel(logging.INFO)

        out = eko.run_dglap(theory, ocard)
        return out

    def run_external(self, theory, ocard, pdf):

        if self.external == "LHA":
            from .external import (  # pylint:disable=import-error,import-outside-toplevel
                LHA_utils,
            )

            # here pdf is not needed
            return LHA_utils.compute_LHA_data(
                theory,
                ocard,
                self.skip_pdfs,
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )
        elif self.external == "LHAPDF":
            from .external import (  # pylint:disable=import-error,import-outside-toplevel
                lhapdf_utils,
            )

            return lhapdf_utils.compute_LHAPDF_data(
                theory,
                ocard,
                pdf,
                self.skip_pdfs,
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )

        elif self.external == "apfel":
            from .external import (  # pylint:disable=import-error,import-outside-toplevel
                apfel_utils,
            )

            return apfel_utils.compute_apfel_data(
                theory,
                ocard,
                pdf,
                self.skip_pdfs,
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )
        else:
            raise NotImplementedError(
                f"Benchmark against {self.external} is not implemented!"
            )

        return {}

    def input_figure(self, theory, ops, pdf_name):
        """
        Pretty-prints the setup to a figure

        Parameters
        ----------
            theory : dict
                theory card
            ops : dict
                operator card
            pdf_name : str
                PDF name

        Returns
        -------
            firstPage : matplotlib.pyplot.Figure
                figure
        """
        firstPage = plt.figure(figsize=(25, 15))
        # theory
        firstPage.text(0.05, 0.97, "Theory:", size=20, ha="left", va="top")
        str_stream = io.StringIO()
        th_copy = theory.copy()
        th_copy.pop("hash", None)
        pprint.pprint(th_copy, stream=str_stream, width=50)
        firstPage.text(0.05, 0.92, str_stream.getvalue(), size=14, ha="left", va="top")
        # operators
        firstPage.text(0.55, 0.87, "Operators:", size=20, ha="left", va="top")
        str_stream = io.StringIO()
        ops_copy = ops.copy()
        ops_copy.pop("hash", None)
        pprint.pprint(ops_copy, stream=str_stream, width=50)
        firstPage.text(0.55, 0.82, str_stream.getvalue(), size=14, ha="left", va="top")
        # pdf
        firstPage.text(0.55, 0.97, "source pdf:", size=20, ha="left", va="top")
        firstPage.text(0.55, 0.92, pdf_name, size=14, ha="left", va="top")
        firstPage.tight_layout()
        return firstPage

    def save_all_operators_to_pdf(self, theory, ops, pdf_name, me):
        """
        Output all operator heatmaps to PDF.

        Parameters
        ----------
            theory : dict
                theory card
            ops : dict
                operator card
            pdf_name : str
                PDF name
            me : eko.output.Output
                DGLAP result
        """

        ops_names = list(me["pids"])
        ops_id = f"o{ops['hash'].hex()[:7]}_t{theory['hash'].hex()[:7]}_{pdf_name}"
        path = f"{self.output_path}/{ops_id}.pdf"
        print(f"Writing operator plots to {ops_id}.pdf")

        with PdfPages(path) as pp:
            # print setup
            firstPage = self.input_figure(theory, ops, pdf_name)
            pp.savefig()
            plt.close(firstPage)
            # print operators
            for q2 in me["Q2grid"].keys():

                ress = me["Q2grid"][q2]["operators"]
                res_errs = me["Q2grid"][q2]["operator_errors"]

                # loop on pids
                for label_out, res, res_err in zip(ops_names, ress, res_errs):

                    if label_out in self.skip_pdfs:
                        continue

                    temp1 = {}
                    err = {}
                    # loop on xgrid point
                    for j in range(len(me["interpolation_xgrid"])):

                        # loop on pid in
                        for label_in, op, op_err in zip(ops_names, res[j], res_err[j]):

                            if label_in in self.skip_pdfs:
                                continue
                            if label_in not in temp1.keys():
                                temp1[label_in] = []
                                err[label_in] = []
                            temp1[label_in].append(op)
                            err[label_in].append(op_err)

                    # temp[label_out] = temp1

                    for label_in in ops_names:

                        if label_in in self.skip_pdfs:
                            continue
                        try:
                            fig = plot_operator(
                                f"Operator ({label_in};{label_out}) µ_F^2 = {q2} GeV^2",
                                temp1[label_in],
                                err[label_in],
                            )
                            pp.savefig()
                        finally:
                            if fig:
                                plt.close(fig)

    def save_final_scale_plots_to_pdf(self, theory, ops, pdf, me, ext):
        """
        Plots all PDFs at the final scale.

        Parameters
        ----------
            theory : dict
                theory card
            ops : dict
                oparator card
            pdf : lhapdf_like
                PDF set
            me : eko.output.Output
                DGLAP result
            ext : dict
                external result
        """
        ref = ext
        ops_id = (
            f"o{ops['hash'].hex()[:7]}_t{theory['hash'].hex()[:7]}_{pdf.set().name}"
        )
        path = f"{self.output_path}/{ops_id}_plots.pdf"
        print(f"Writing pdf plots to {ops_id}_plots.pdf")

        xgrid = ref["target_xgrid"]
        q2s = list(ext["values"].keys())
        pdf_grid = me.apply_pdf(
            pdf,
            xgrid,
            rotate_to_evolution_basis=self.rotate_to_evolution_basis,
        )

        with PdfPages(path) as pp:

            # print setup
            firstPage = self.input_figure(theory, ops, pdf.set().name)
            pp.savefig()
            plt.close(firstPage)

            # iterate all pdf
            for q2 in q2s:
                res = pdf_grid[q2]
                my_pdfs = res["pdfs"]
                my_pdf_errs = res["errors"]
                ref_pdfs = ext["values"][q2]

                for key in my_pdfs:
                    # skip trivial plots
                    if key in self.skip_pdfs:
                        continue
                    # plot
                    fig = plot_dist(
                        xgrid,
                        xgrid * my_pdfs[key],
                        xgrid * my_pdf_errs[key],
                        ref_pdfs[key],
                        title=f"x{pdfname(key)}(x,µ_F^2 = {q2} GeV^2)",
                    )
                    pp.savefig()
                    plt.close(fig)

    def log(self, theory, ocard, pdf, me, ext):

        pdf_name = pdf.set().name

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # TODO: do we want to keep this? run.me and only for sandbox.
        # dump operators to file
        if self.post_process_config["write_operator"]:
            ops_id = (
                f"o{ocard['hash'].hex()[:7]}_t{theory['hash'].hex()[:7]}_{pdf_name}"
            )
            path = f"{self.output_path}/{ops_id}.yaml"
            print(f"Writing operator to {ops_id}.yaml")
            me.dump_yaml_to_file(path, binarize=False)

        # TODO: do we want to keep this? this goes to navigator.
        # pdf comparison
        if self.post_process_config["plot_PDF"]:
            self.save_final_scale_plots_to_pdf(theory, ocard, pdf, me, ext)

        # TODO: do we want to keep this? this to navigator as well.
        # graphical representation of operators
        if self.post_process_config["plot_operator"]:
            self.save_all_operators_to_pdf(theory, ocard, pdf_name, me)

        # return a proper log table
        log_tabs = {}
        xgrid = ext["target_xgrid"]
        q2s = list(ext["values"].keys())
        pdf_grid = me.apply_pdf(
            pdf,
            xgrid,
            rotate_to_evolution_basis=self.rotate_to_evolution_basis,
        )
        for q2 in q2s:

            log_tab = dfdict.DFdict()
            ref_pdfs = ext["values"][q2]
            res = pdf_grid[q2]
            my_pdfs = res["pdfs"]
            my_pdf_errs = res["errors"]

            for key in my_pdfs:

                if key in self.skip_pdfs:
                    continue

                # build table
                tab = {}
                tab["x"] = xgrid
                tab["eko"] = f = xgrid * my_pdfs[key]
                tab["eko_error"] = xgrid * my_pdf_errs[key]
                tab[self.external] = r = ref_pdfs[key]
                tab["percent_error"] = (f - r) / r * 100

                tab = pd.DataFrame(tab)
                log_tab[key] = tab
            log_tabs[q2] = log_tab

        return log_tabs
