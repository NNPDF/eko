"""EKO implementation of navigator."""

import os
import webbrowser

import matplotlib.pyplot as plt
import pandas as pd
from banana import cfg as banana_cfg
from banana import navigator as bnav
from banana.data import dfdict
from matplotlib.backends.backend_pdf import PdfPages

from eko import basis_rotation as br

from .. import pdfname
from ..data import db
from ..plots import input_figure, plot_dist

table_objects = bnav.navigator.table_objects
table_objects["o"] = db.Operator


class NavigatorApp(bnav.navigator.NavigatorApp):
    """Navigator base class holding all elementry operations.

    Parameters
    ----------
    cfg : dict
        banana configuration
    mode : str
        mode identifier
    """

    myname = "eko"
    table_objects = table_objects

    def fill_theories(self, theo, obj):
        """Collect important information of the theory record.

        Parameters
        ----------
        theo : dict
            database record
        obj : dict
            to be updated pandas record
        """
        for f in [
            "PTO",
            "ModEv",
            "Q0",
            "Qref",
            "alphas",
            "XIF",
        ]:
            obj[f] = theo[f]
        obj["XIF"] = theo["XIF"]
        obj["mcThr"] = theo["mc"] * theo["kcThr"]
        obj["mbThr"] = theo["mb"] * theo["kbThr"]
        obj["mtThr"] = theo["mt"] * theo["ktThr"]

    def fill_operators(self, op, obj):
        """Collect important information of the operator record.

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
        obj["mugrid"] = op["mugrid"]
        obj["max_ord"] = op["ev_op_max_order"]
        obj["iters"] = op["ev_op_iterations"]
        obj["skip_ns"] = op["debug_skip_non_singlet"]
        obj["skip_s"] = op["debug_skip_singlet"]
        obj["pol"] = op["polarized"]
        obj["time"] = op["time_like"]

    def fill_cache(self, cac, obj):
        """Collect important information of the cache record.

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
        """Collect important information of the log record.

        Parameters
        ----------
        lg : dict
            database record
        obj : dict
            to be updated pandas record
        """
        q2s = lg["log"].q2s
        crash = lg.get("_crash", None)
        if crash is None:
            obj["q2s"] = f"{q2s}"
        else:
            obj["q2s"] = crash

        obj["theory"] = lg["t_hash"][: self.hash_len]
        obj["ocard"] = lg["o_hash"][: self.hash_len]
        for f in ["pdf", "external"]:
            obj[f] = lg[f]

    def check_log(self, doc_hash, perc_thr=1, abs_thr=1e-6):
        """Check if the log passed the default assertions.

        Parameters
        ----------
        doc_hash : hash
            log hash
        """
        dfds = self.log_as_dfd(doc_hash)
        log = self.get(bnav.l, doc_hash)

        for q2 in dfds:
            for op, df in dfds[q2].items():
                for row in df.iloc:
                    if (
                        abs(row["percent_error"]) > perc_thr
                        and abs(row[f"{log['external']}"] - row["eko"]) > abs_thr
                    ):
                        print(op, row, sep="\n", end="\n\n")

    def plot_pdfs(self, doc_hash):
        """Plot all PDFs at the final scale.

        Parameters
        ----------
        doc_hash : hash
            log hash
        """
        log = self.get(bnav.l, doc_hash)
        dfd = log["log"]
        directory = (
            banana_cfg.cfg["database_path"].parents[0] / f"{log['external']}_bench"
        )

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
        """Display PDF generated by ekomark.navigator.navigator.plot_pdfs.

        Parameters
        ----------
        doc_hash : hash
            log hash
        """
        log = self.get(bnav.l, doc_hash)
        directory = (
            banana_cfg.cfg["database_path"].parents[0] / f"{log['external']}_bench"
        )

        if not directory.exists():
            directory.mkdir(parents=True)

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
        """Check if is actual name.

        Parameters
        ----------
        name : str
            name

        Returns
        -------
        bool :
            is actual name?
        """
        return name in br.evol_basis or name in br.flavor_basis_names

    def compare_external(self, dfd1, dfd2):
        """Compare two results in the cache.

        It's taking two results from external benchmarks and compare them in a
        single table.

        Parameters
        ----------
        dfd1 : dict or hash
            if hash the doc_hash of the cache to be loaded
        dfd2 : dict or hash
            if hash the doc_hash of the cache to be loaded
        """
        # load json documents
        id1, cache1 = self.load_dfd(dfd1, self.cache_as_dfd)
        id2, cache2 = self.load_dfd(dfd2, self.cache_as_dfd)

        if cache1.external == cache2.external:
            cache1.external = f"{cache1.external}1"
            cache2.external = f"{cache2.external}2"

        # print head
        cache_diff = dfdict.DFdict()
        msg = f"**Comparing** id: `{id1}` - id: `{id2}`, in table *cache*"
        cache_diff.print(msg, "-" * len(msg), sep="\n")
        cache_diff.print(f"- *{cache1.external}*: `{id1}`")
        cache_diff.print(f"- *{cache2.external}*: `{id2}`")
        cache_diff.print()

        # check x
        for x in cache1["target_xgrid"]:
            if x not in cache2["target_xgrid"]:
                raise ValueError(f"{x}: not matching in x")

        table_out = dfdict.DFdict()
        for q2, pdfs1 in cache1["values"].items():
            # check q2
            if q2 not in cache2["values"].keys():
                raise ValueError(f"{q2}: not matching in q2")

            pdfs2 = cache2["values"][q2]
            # check pdfs
            for pdf in pdfs1.T.keys():
                if pdf not in pdfs2.keys():
                    raise ValueError(f"{pdf}: pdf not matching")
                # skip the photon always (just for the time being)
                if pdf == "ph":
                    continue

                tab = {
                    "x": cache1["target_xgrid"][0],
                    cache1.external: pdfs1[pdf],
                    cache2.external: pdfs2[pdf],
                    "percent_error": (pdfs1[pdf] / pdfs2[pdf] - 1.0) * 100,
                }
                table_out[pdf] = pd.DataFrame.from_dict(tab)

            # dump results' table
            cache_diff[q2] = table_out

        return cache_diff
