# -*- coding: utf-8 -*-
import os
import webbrowser

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from banana import navigator as bnav

from eko import basis_rotation as br

from .. import pdfname
from ..plots import input_figure, plot_dist
from ..banana_cfg import banana_cfg
from ..data import db

table_objects = bnav.table_objects
table_objects["o"] = db.Operator


class NavigatorApp(bnav.navigator.NavigatorApp):
    """
    Navigator base class holding all elementry operations.

    Parameters
    ----------
        cfg : dict
            banana configuration
        mode : string
            mode identifier
    """

    myname = "eko"
    table_objects = table_objects

    def fill_theories(self, theo, obj):
        """
        Collect important information of the theory record.

        Parameters
        ----------
            theo : dict
                database record
            obj : dict
                to be updated pandas record
        """
        for f in [
            "PTO",
            "XIR",
            "ModEv",
            "Q0",
            "Qref",
            "alphas",
        ]:
            obj[f] = theo[f]
        obj["kcThr_mc"] = theo["mc"] * theo["kcThr"]
        obj["kbThr_mb"] = theo["mb"] * theo["kbThr"]
        obj["ktThr_mt"] = theo["mt"] * theo["ktThr"]

    def fill_operators(self, op, obj):
        """
        Collect important information of the operator record.

        Parameters
        ----------
            op : dict
                database record
            obj : dict
                to be updated pandas record
        """
        xgrid = op["interpolation_xgrid"]
        obj["xgrid"] = (
            f"{len(xgrid)}pts: "
            + f"{'log' if op['interpolation_is_log'] else 'x'}"
            + f"^{op['interpolation_polynomial_degree']}"
        )

        obj["debug_skip_non_singlet"] = op["debug_skip_non_singlet"]
        obj["ev_op_max_order"] = op["ev_op_max_order"]
        obj["ev_op_iterations"] = op["ev_op_iterations"]
        obj["Q2grid"] = op["Q2grid"]

    def fill_cache(self, cac, obj):
        """
        Collect important information of the cache record.

        Parameters
        ----------
            cac : dict
                database record
            obj : dict
                to be updated pandas record
        """
        vals = cac["result"]["values"]
        q2s = list(vals.keys())
        # assume the vals are homogenous (true for bare eko results) and look
        # only at the first one
        pdfs = len(next(iter(vals.values())))

        obj["operators"] = f"{pdfs} pdfs @ Q^2 {q2s} Gev^2"

        obj["theory"] = cac["t_hash"][: self.hash_len]
        obj["observables"] = cac["o_hash"][: self.hash_len]
        for f in ["pdf", "external"]:
            obj[f] = cac[f]

    def fill_logs(self, lg, obj):
        """
        Collect important information of the log record.

        Parameters
        ----------
            lg : dict
                database record
            obj : dict
                to be updated pandas record
        """
        q2s = lg["log"].q2s
        pdfs = len(lg["log"])
        crash = lg.get("_crash", None)
        if crash is None:
            obj["operators"] = f"{pdfs}  pdfs @ Q^2 {q2s} Gev^2"
        else:
            obj["operators"] = crash

        obj["theory"] = lg["t_hash"][: self.hash_len]
        obj["observables"] = lg["o_hash"][: self.hash_len]
        for f in ["pdf", "external"]:
            obj[f] = lg[f]

    def check_log(self, doc_hash, perc_thr=1, abs_thr=1e-6):
        """
        Check if the log passed the default assertions

        Parameters
        ----------
            doc_hash : hash
                log hash
        """
        dfds = self.log_as_dfd(doc_hash)
        log = self.get(bnav.l, doc_hash)

        for q2 in dfds:
            for op, df in dfds[q2].items():
                for l in df.iloc:
                    if (
                        abs(l["percent_error"]) > perc_thr
                        and abs(l[f"{log['external']}"] - l["eko"]) > abs_thr
                    ):
                        print(op, l, sep="\n", end="\n\n")

    def plot_pdfs(self, doc_hash):
        """
        Plots all PDFs at the final scale.

        Parameters
        ----------
            doc_hash : hash
                log hash
        """
        log = self.get(bnav.l, doc_hash)
        dfd = log["log"]
        directory = banana_cfg["database_path"].parents[0] / f"{log['external']}_bench"

        if not os.path.exists(directory):
            os.makedirs(directory)

        ops_id = log["hash"][: self.hash_len]
        path = directory / f"{ops_id}.pdf"
        print(f"Writing pdf plots to {path}")

        with PdfPages(path) as pp:

            # print setup
            theory = self.get(bnav.t, log["t_hash"][: self.hash_len])
            ops = self.get(bnav.o, log["o_hash"][: self.hash_len])
            firstPage = input_figure(theory, ops, pdf_name=log["pdf"])
            pp.savefig()
            plt.close(firstPage)

            # iterate all pdf
            for q2 in dfd.q2s:
                for op, key in dfd.q2_slice(q2).items():
                    # plot
                    fig = plot_dist(
                        key["x"],
                        key["eko"],
                        key["eko_error"],
                        key[f"{log['external']}"],
                        title=f"x{pdfname(op)}(x,µ_F^2 = {q2} GeV^2)",
                    )
                    pp.savefig()
                    plt.close(fig)

        return path

    def display_pdfs(self, doc_hash):
        """
        Display PDF generated by ekomark.navigator.navigator.plot_pdfs

        Parameters
        ----------
            doc_hash : hash
                log hash
        """
        log = self.get(bnav.l, doc_hash)
        directory = banana_cfg["database_path"].parents[0] / f"{log['external']}_bench"

        path = None
        for plot_file in directory.iterdir():
            if plot_file.stem[: len(doc_hash)] == doc_hash:
                path = plot_file
                break

        if path is None:
            print("Drawing the plots...")
            path = self.plot_pdfs(doc_hash)

        webbrowser.open(str(path.absolute()))

    @staticmethod
    def is_valid_physical_object(name):
        return name in br.evol_basis or name in br.flavor_basis_names
