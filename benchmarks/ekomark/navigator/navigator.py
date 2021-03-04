# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from banana import navigator as bnav
from banana.data import dfdict

from ekomark.plots import input_figure, plot_dist
from ekomark.banana_cfg import banana_cfg

from eko import basis_rotation as br

from ..data import db

table_objects = bnav.table_objects
table_objects["o"] = db.Operator


def pdfname(pid_or_name):
    """ Return pdf name  """
    if isinstance(pid_or_name, int):
        return br.flavor_basis_names[br.flavor_basis_pids.index(pid_or_name)]
    return pid_or_name


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

    def subtract_tables(self, hash1, hash2):
        """
        Subtract results in the second table from the first one,
        properly propagate the integration error and recompute the relative
        error on the subtracted results.

        Parameters
        ----------
            hash1 : hash
                if hash the doc_hash of the log to be loaded
            hash2 : hash
                if hash the doc_hash of the log to be loaded

        Returns
        -------
            diffsout : DFdict
                created frames
        """

        diffsout = dfdict.DFdict()

        # load json documents
        logs = []
        ids = []
        for h in [hash1, hash2]:
            logs.append(self.log_as_dfd(h))
            ids.append(h)

        log1, log2 = logs
        id1, id2 = ids

        # print head
        msg = f"Subtracting id: '{id1}' - id: '{id2}', in table 'logs'"
        diffsout.print(msg, "=" * len(msg), sep="\n")
        diffsout.print()

        if log1 is None:
            raise ValueError(f"Log id: '{id1}' not found")
        if log2 is None:
            raise ValueError(f"Log id: '{id2}' not found")

        # iterate operators
        for q2 in log1.keys():
            if q2 not in log2.keys():
                print(f"{q2}: not matching in log2")
                continue

            diffout = dfdict.DFdict()
            for op, tab in log1[q2].items():
                # load operators tables
                table1 = pd.DataFrame(tab)
                table2 = pd.DataFrame(log2[q2][op])
                table_out = table2.copy()

                # check for compatible kinematics
                if any([any(table1[y] != table2[y]) for y in ["x"]]):
                    raise ValueError("Cannot compare tables with different (x)")

                # subtract and propagate
                known_col_set = set(["x", "eko", "eko_error", "percent_error"])
                t1_ext = list(set(table1.keys()) - known_col_set)[0]
                t2_ext = list(set(table2.keys()) - known_col_set)[0]
                if t1_ext == t2_ext:
                    tout_ext = t1_ext
                else:
                    tout_ext = f"{t2_ext}-{t1_ext}"
                table_out.rename(columns={t2_ext: tout_ext}, inplace=True)
                table_out[tout_ext] = table2[t2_ext] - table1[t1_ext]
                # subtract our values
                table_out["eko"] -= table1["eko"]
                table_out["eko_error"] += table1["eko_error"]

                # compute relative error
                def rel_err(row, tout_ext=tout_ext):
                    if row[tout_ext] == 0.0:
                        if row["eko"] == 0.0:
                            return 0.0
                        return np.nan
                    else:
                        return (row["eko"] / row[tout_ext] - 1.0) * 100

                table_out["percent_error"] = table_out.apply(rel_err, axis=1)

                # dump results' table
                diffout.print(op, "-" * len(op), sep="\n")
                diffout[op] = table_out
            diffsout[q2] = diffout
        return diffsout

    def check_log(self, doc_hash, perc_thr=1, abs_thr=1e-6):
        """
        Check if the log passed the default assertions

        Paramters
        ---------
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
        path = f"{banana_cfg['database_path'].parents[0]}/{log['external']}_bench"

        if not os.path.exists(path):
            os.makedirs(path)

        ops_id = f"{log['hash'][:self.hash_len]}"
        path = f"{path}/{ops_id}.pdf"
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

    @staticmethod
    def is_valid_physical_object(name):
        return name in br.evol_basis or name in br.flavor_basis_names

    # def join(self, id1, id2):
    #     tabs = []
    #     tabs1 = []
    #     exts = []
    #     suffixes = (f" ({id1})", f" ({id2})")

    #     for i, doc_hash in enumerate([id1, id2]):
    #         tabs += [self.get_log_DFdict(doc_hash)[0]]
    #         tabs1 += [tabs[i].drop(["yadism", "yadism_error", "percent_error"], axis=1)]
    #         exts += [
    #             tabs1[i].columns.drop(["x", "Q2"])[0]
    #         ]  # + suffixes[i]] # to do: the suffixes are not working as expected

    #     def rel_err(row):
    #         ref = row[exts[0]]
    #         cmp = row[exts[1]]
    #         if ref != 0:
    #             return (cmp / ref - 1) * 100
    #         else:
    #             return np.nan

    #     tab_joint = tabs1[0].merge(
    #         tabs1[1], on=["x", "Q2"], how="outer", suffixes=suffixes
    #     )
    #     tab_joint["ext_rel_err [%]"] = tab_joint.apply(rel_err, axis=1)

    #     if all(np.isclose(tabs[0]["yadism"], tabs[1]["yadism"])):
    #         tab_joint["yadism"] = tabs[0]["yadism"]
    #         tab_joint["yadism_error"] = tabs[0]["yadism_error"]
    #     else:
    #         pass

    #     return tab_joint
