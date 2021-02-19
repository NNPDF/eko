# -*- coding: utf-8 -*-
"""
Abstract layer for running the benchmarks
"""

import pathlib
import pprint
import io
import abc

import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import lhapdf

import eko

from . import toyLH
from .plots import plot_dist, plot_operator


def pdfname(pid_or_name):
    if isinstance(pid_or_name, int):
        return eko.basis_rotation.flavor_basis_names[
            eko.basis_rotation.flavor_basis_pids.index(pid_or_name)
        ]
    return pid_or_name


class Runner(abc.ABC):
    """
    Abstract layer for running the benchmarks

    Parameters
    ----------
        theory_path : string or pathlib.Path
            path to theory card
        operators_path : string or pathlib.Path
            path to operators card
        assets_dir : string
            output directory
    """

    def __init__(self, theory_path, operators_path, assets_dir):
        self.output_path = ""

        # read both cards
        cards = []
        for path in [theory_path, operators_path]:
            if not isinstance(path, pathlib.Path):
                path = pathlib.Path(path)

            self.output_path += path.stem + "_"

            with open(path, "r") as infile:
                cards.append(yaml.safe_load(infile))
        self.theory = cards[0]
        self.operators = cards[1]

        # output dir
        self.assets_dir = assets_dir
        # default config for post processing
        self.post_process_config = {
            "plot_PDF": True,
            "plot_operator": False,  # True,
            "write_operator": True,
        }

    def _post_process(self, output):
        """
        Handles the post processing of the run according to the configuration.

        Parameters
        ----------
            output : eko.output.Output
                EKO result
            ref : dict
                reference result
            tag : string
                file tag
        """
        # dump operators to file
        if self.post_process_config["write_operator"]:
            p = self.assets_dir / (self.output_path + "-ops.yaml")
            output.dump_yaml_to_file(p)
            print(f"write operator to {p}")
        # pdf comparison
        if self.post_process_config["plot_PDF"]:
            self.save_final_scale_plots_to_pdf(output)
        # graphical representation of operators
        if self.post_process_config["plot_operator"]:
            p = self.assets_dir / (self.output_path + "-ops.pdf")
            self.save_all_operators_to_pdf(output, p)
            print(f"write operator plots to {p}")

    def run(self):
        """
        Runs the input card

        Returns
        -------
            ret : dict
                DGLAP result
        """
        # check cache
        target_path = self.assets_dir / (self.output_path + "-ops.yaml")
        rerun = True
        if target_path.exists():
            rerun = False
            ask = input("Use cached output? [Y/n]")
            if ask.lower() in ["n", "no"]:
                rerun = True
        if rerun:
            ret = eko.run_dglap(self.theory, self.operators)
        else:
            # load
            self.post_process_config["write_operator"] = False
            with open(target_path) as o:
                ret = eko.output.Output.load_yaml(o)
        self._post_process(ret)
        return ret

    def input_figure(self, pdf_name):
        """
        Pretty-prints the setup to a figure

        Parameters
        ----------
            output : eko.output.Output
                DGLAP result

        Returns
        -------
            firstPage : matplotlib.pyplot.Figure
                figure
        """
        firstPage = plt.figure(figsize=(13, 10))
        # theory
        firstPage.text(0.05, 0.97, "Theory:", size=20, ha="left", va="top")
        str_stream = io.StringIO()
        pprint.pprint(self.theory, stream=str_stream, width=50)
        firstPage.text(0.05, 0.92, str_stream.getvalue(), size=14, ha="left", va="top")
        # operators
        firstPage.text(0.55, 0.97, "Operators:", size=20, ha="left", va="top")
        str_stream = io.StringIO()
        pprint.pprint(self.operators, stream=str_stream, width=50)
        firstPage.text(0.55, 0.92, str_stream.getvalue(), size=14, ha="left", va="top")
        # pdf
        firstPage.text(0.55, 0.47, "source pdf:", size=20, ha="left", va="top")
        firstPage.text(0.55, 0.42, pdf_name, size=14, ha="left", va="top")
        return firstPage

    def save_all_operators_to_pdf(self, output, path):
        """
        Output all operator heatmaps to PDF.

        Parameters
        ----------
            output : eko.output.Output
                DGLAP result
            path : string
                target file name
        """
        first_ops = list(output["Q2grid"].values())[0]
        with PdfPages(path) as pp:
            # print setup
            firstPage = self.input_figure("")
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

    @abc.abstractmethod
    def ref(self):
        pass

    def save_final_scale_plots_to_pdf(self, output):
        """
        Plots all PDFs at the final scale.

        Parameters
        ----------
            path : string
                output path
            output : eko.output.Output
                DGLAP result
        """
        ref = self.ref()
        path = self.assets_dir / (
            self.output_path + f"-{ref['src_pdf']}" + "-plots.pdf"
        )
        print(f"write pdf plots to {path}")
        xgrid = ref["target_xgrid"]
        first_q2 = list(ref["values"].keys())[0]
        ref_pdfs = list(ref["values"].values())[0]
        # get my data
        if ref["src_pdf"] == "ToyLH":
            pdf = toyLH.mkPDF("", "")
        else:
            pdf = lhapdf.mkPDF(ref["src_pdf"])
        pdf_grid = output.apply_pdf(
            pdf, xgrid, rotate_to_evolution_basis=ref["rotate_to_evolution_basis"],
        )
        first_res = list(pdf_grid.values())[0]
        my_pdfs = first_res["pdfs"]
        my_pdf_errs = first_res["errors"]
        with PdfPages(path) as pp:
            # print setup
            firstPage = self.input_figure(ref["src_pdf"])
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
