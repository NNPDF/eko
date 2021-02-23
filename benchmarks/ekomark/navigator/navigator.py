# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from banana import navigator as bnav
from banana.data import dfdict

from ekomark.plots import input_figure, plot_dist
from ekomark.banana_cfg import banana_cfg


def pdfname(pid_or_name):
    """ Return pdf name  """
    if isinstance(pid_or_name, int):
        return eko.basis_rotation.flavor_basis_names[
            eko.basis_rotation.flavor_basis_pids.index(pid_or_name)
        ]
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
            "NfFF",
            "FNS",
            "ModEv",
            "Q0",
            "kcThr",
            "kbThr",
            "ktThr",
            "mc",
            "mb",
            "mt",
            "Qref",
            "alphas",
        ]:
            obj[f] = theo[f]

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
        for f in ["external", "pdf"]:
            obj[f] = cac[f]

        q2s = []
        ops = 0
        vals = cac["result"]["values"]

        for q2 in list(vals.keys()):

            q2s.append(q2)
            # all the q2s have the same number of operators
            ops = len(vals[q2])

        obj["operators"] = f"{ops} Operators @ Q^2 {q2s} Gev^2"

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

        q2s = []
        ops = 0
        for q2 in list(lg["log"].keys()):

            q2s.append(q2)
            # all the q2s have the same number of operators
            ops = len(lg["log"][q2].values())

        crash = lg.get("_crash", None)
        if crash is None:
            obj["operators"] = f"{ops}  Operators @ Q^2 {q2s} Gev^2"
        else:
            obj["operators"] = crash

        for f in ["external", "pdf"]:
            obj[f] = lg[f]

    def list_all_similar_logs(self, ref_hash):
        """
        Search logs which are similar to the one given, i.e., same theory and,
        same operators, and same pdfset.

        Parameters
        ----------
            ref_hash : hash
                partial hash of the reference log

        Returns
        -------
            df : pandas.DataFrame
                created frame

        Note
        ----
        The external it's not used to discriminate logs: even different
        externals should return the same numbers, so it's relevant to keep all
        of them.
        """
        # obtain reference log
        ref_log = self.get(bnav.l, ref_hash)

        related_logs = []
        all_logs = self.get(bnav.l)

        for lg in all_logs:
            if lg["t_hash"] != ref_log["t_hash"]:
                continue
            if lg["o_hash"] != ref_log["o_hash"]:
                continue
            if lg["pdf"] != ref_log["pdf"]:
                continue
            related_logs.append(lg)

        return self.list_all(bnav.l, related_logs)

    def log_as_dfd(self, doc_hash):
        """
        Load all structure functions in log as dict of DataFrames

        Parameters
        ----------
            doc_hash : hash
                document hash

        Returns
        -------
            log : DFdict
                DataFrames
        """
        log = self.get(bnav.l, doc_hash)

        q2s = []
        dfds = dfdict.DFdict()
        for q2 in list(log["log"].keys()):

            dfd = dfdict.DFdict()
            dfd.print(
                f"Q^2 {q2}"
                + f"with theory={log['t_hash']}, "
                + f"operators={log['o_hash']} "
                + f"using {log['pdf']}"
            )
            for op, tab in log["log"][q2].items():
                dfd.print(f"Operator {op}")
                dfd[op] = pd.DataFrame(tab)
            dfds[q2] = dfd
        return dfds

    # TODO: test this
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

    def crashed_log(self, doc_hash):
        """
        Check if the log passed the default assertions

        Paramters
        ---------
            doc_hash : hash
                log hash

        Returns
        -------
            cdfd : dict
                log without kinematics
        """

        dfds = self.log_as_dfd(doc_hash)
        if "_crash" not in dfds:
            raise ValueError("log didn't crash!")

        cdfds = {}
        for q2 in dfds:
            cdfd = dfds[q2].copy()
            for op, df in dfds[q2].items():
                cdfd[op].pop("x")
            cdfds[q2] = cdfd

        return cdfds

    def plot_pdfs(self, doc_hash):
        """
        Plots all PDFs at the final scale.

        Parameters
        ----------
            doc_hash : hash
                log hash
        """

        dfds = self.log_as_dfd(doc_hash)
        log = self.get(bnav.l, doc_hash)
        path = f"{banana_cfg['database_path'].parents[0]}/{log['external']}_bench"

        if not os.path.exists(path):
            os.makedirs(path)

        ops_id = f"o{log['o_hash'].hex()[:6]}_t{log['t_hash'].hex()[:6]}_{log['pdf']}"
        path = f"{path}/{ops_id}_plots.pdf"
        print(f"Writing pdf plots to {path}")

        with PdfPages(path) as pp:

            # print setup
            theory = self.get(bnav.t, log["t_hash"].hex()[:6])
            ops = self.get(bnav.o, log["o_hash"].hex()[:6])
            firstPage = input_figure(theory, ops, pdf_name=log["pdf"])
            pp.savefig()
            plt.close(firstPage)

            # iterate all pdf
            for q2 in dfds:
                for op, key in dfds[q2].items():
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
