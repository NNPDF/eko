# -*- coding: utf-8 -*-
"""
Abstract layer for running the benchmarks
"""
import pprint
import io
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
    if isinstance(pid_or_name, int):
        return eko.basis_rotation.flavor_basis_names[
            eko.basis_rotation.flavor_basis_pids.index(pid_or_name)
        ]
    return pid_or_name


class Runner(BenchmarkRunner):
    banana_cfg = banana_cfg

    @staticmethod
    def init_ocards(conn):
        """
        Create opertors card

        Parameters
        ----------
            conn : sqlite3.Connection
                DB connection

        Returns
        -------

        """
        with conn:
            conn.execute(sql.create_table("operators", operators.default_card))

    @staticmethod
    def load_ocards(conn, ocard_updates):
        """
        Load o-cards from the DB.

        Parameters
        ----------
            conn : sqlite3.Connection
                DB connection
            ocard_updates : list(dict)
                o-card configurations

        Returns
        -------
            ocards : list(dict)
                all requested o-cards
        """
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

        # TODO: check cache using banana ?
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

        out = eko.run_dglap(theory, ocard)
        return out

    def run_external(self, theory, ocard, pdf):
        """
        Run external program: LHA, LHAPDF, apfel

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
            any : dict 
                external dict results 
        """        

        # TODO: add other theory checks, if necessary

        if self.external == "LHA":
            from .external import (  # pylint:disable=import-error,import-outside-toplevel
                LHA_utils,
            )

            return LHA_utils.compute_LHA_data(
                theory, ocard, pdf, rotate_to_evolution_basis=self.rtevb
            )

        if self.external == "LHAPDF":
            from .external import (  # pylint:disable=import-error,import-outside-toplevel
                lhapdf_utils,
            )

            return lhapdf_utils.compute_LHAPDF_data(
                theory, ocard, pdf, rotate_to_evolution_basis=self.rtevb
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
        firstPage = plt.figure(figsize=(13, 10))
        # theory
        firstPage.text(0.05, 0.97, "Theory:", size=20, ha="left", va="top")
        str_stream = io.StringIO()
        pprint.pprint(theory, stream=str_stream, width=50)
        firstPage.text(0.05, 0.92, str_stream.getvalue(), size=14, ha="left", va="top")
        # operators
        firstPage.text(0.55, 0.97, "Operators:", size=20, ha="left", va="top")
        str_stream = io.StringIO()
        pprint.pprint(ops, stream=str_stream, width=50)
        firstPage.text(0.55, 0.92, str_stream.getvalue(), size=14, ha="left", va="top")
        # pdf
        firstPage.text(0.55, 0.47, "source pdf:", size=20, ha="left", va="top")
        firstPage.text(0.55, 0.42, pdf_name, size=14, ha="left", va="top")
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
        first_ops = list(me["Q2grid"].values())[0]
        ops_id = f"o{ops['hash'].hex()[:7]}_t{theory['hash'].hex()[:7]}_{pdf_name}"
        path = f"{self.output_path}/{ops_id}.pdf"
        print(f"Writing operator plots to {ops_id}.pdf")

        with PdfPages(path) as pp:
            # print setup
            firstPage = self.input_figure(theory, ops, pdf_name)
            pp.savefig()
            plt.close(firstPage)
            # print operators
            for label in first_ops["operators"]:
                try:
                    fig = plot_operator(first_ops, label)
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
        ops_id = f"o{ops['hash'].hex()[:7]}_t{theory['hash'].hex()[:7]}_{pdf.set().name}"
        path = f"{self.output_path}/{ops_id}_plots.pdf"
        print(f"Writing pdf plots to {ops_id}_plots.pdf")

        xgrid = ref["target_xgrid"]
        first_q2 = list(ref["values"].keys())[0]
        ref_pdfs = list(ref["values"].values())[0]

        pdf_grid = me.apply_pdf(
            pdf, xgrid, rotate_to_evolution_basis=ref["rotate_to_evolution_basis"],
        )
        first_res = list(pdf_grid.values())[0]
        my_pdfs = first_res["pdfs"]
        my_pdf_errs = first_res["errors"]
        with PdfPages(path) as pp:
            # print setup
            firstPage = self.input_figure(theory, ops, pdf.set().name)
            pp.savefig()
            plt.close(firstPage)
            # iterate all pdf
            for key in my_pdfs:
                # skip trivial plots
                if key in ref["skip_pdfs"]:
                    continue
                # plot
                fig = plot_dist(
                    xgrid,
                    xgrid * my_pdfs[key],
                    xgrid * my_pdf_errs[key],
                    ref_pdfs[key],
                    title=f"x{pdfname(key)}(x,µ_F^2 = {first_q2} GeV^2)",
                )
                pp.savefig()
                plt.close(fig)

    def log(self, theory, ocard, pdf, me, ext):
        """
        Handles the post processing of the run according to the configuration.
        
        Parameters
        ----------
            theory : dict
                theory card
            ocard : dict
                operator card
            pdf : lhapdf_like
                PDF set
            me : eko.output.Output
                DGLAP result
            ext : dict
                external result
        Returns
        -------
            log_tab: dfdict 
                Log table
        """

        pdf_name = pdf.set().name

        # TODO: do we want to keep this?
        # dump operators to file
        if self.post_process_config["write_operator"]:
            ops_id = f"o{ocard['hash'].hex()[:7]}_t{theory['hash'].hex()[:7]}_{pdf_name}"
            path = f"{self.output_path}/{ops_id}.yaml"
            print(f"Writing operator to {ops_id}.yaml")
            me.dump_yaml_to_file(path, binarize=False)   

        # TODO: do we want to keep this?
        # pdf comparison
        if self.post_process_config["plot_PDF"]:
            self.save_final_scale_plots_to_pdf(theory, ocard, pdf, me, ext)

        # TODO: do we want to keep this?
        # graphical representation of operators
        if self.post_process_config["plot_operator"]:
            self.save_all_operators_to_pdf(theory, ocard, pdf_name, me)

        # return a proper log table
        log_tab = dfdict.DFdict()
        xgrid = ext["target_xgrid"]
        first_q2 = list(ext["values"].keys())[0]
        ref_pdfs = list(ext["values"].values())[0]
        pdf_grid = me.apply_pdf(
            pdf, xgrid, rotate_to_evolution_basis=ext["rotate_to_evolution_basis"],
        )
        first_res = list(pdf_grid.values())[0]
        my_pdfs = first_res["pdfs"]
        my_pdf_errs = first_res["errors"]

        tabs = []
        for key in my_pdfs:
            # skip trivial plots
            if key in ext["skip_pdfs"]:
                continue

            # build table
            tab = {}
            tab["x"] = xgrid
            tab["Eko pdf"] = f = xgrid * my_pdfs[key]
            tab["Eko err"] = xgrid * my_pdf_errs[key]
            tab[self.external] = r = ref_pdfs[key]
            tab["percent_error"] = (f - r) / r * 100

            tab = pd.DataFrame(tab)
            log_tab[key] = tab

        return log_tab
