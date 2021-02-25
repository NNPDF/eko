# -*- coding: utf-8 -*-
"""
Abstract layer for running the benchmarks
"""
import os
import logging
import sys
import pandas as pd

from banana.data import sql, dfdict
from banana.benchmark.runner import BenchmarkRunner

from ekomark.banana_cfg import banana_cfg
from ekomark.data import operators, db
import eko


class Runner(BenchmarkRunner):
    banana_cfg = banana_cfg
    db_base_class = db.Base
    rotate_to_evolution_basis = False
    skip_pdfs = []
    sandbox = False

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
            pdf : lhapdf_type
                pdf

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

        # if sandbox check for cache, dump eko to yaml
        # and plot the operators
        if self.sandbox:
            rerun = True
            ops_id = f"o{ocard['hash'].hex()[:6]}_t{theory['hash'].hex()[:6]}"
            path = f"{banana_cfg['database_path'].parents[0]}/{ops_id}.yaml"

            if os.path.exists(path):
                rerun = False
                # ask = input("Use cached output? [Y/n]")
                # if ask.lower() in ["n", "no"]:
                #     rerun = True

            if rerun:
                out = eko.run_dglap(theory, ocard)
                print(f"Writing operator to {path}")
                out.dump_yaml_to_file(path, binarize=False)
            else:
                # load
                print("Using cached eko data")
                with open(path) as o:
                    out = eko.output.Output.load_yaml(o)

            if self.plot_operator:
                from ekomark.plots import (  # pylint:disable=import-error,import-outside-toplevel
                    save_operators_to_pdf,
                )

                output_path = (
                    f"{banana_cfg['database_path'].parents[0]}/{self.external}_bench"
                )
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                save_operators_to_pdf(output_path, theory, ocard, out, self.skip_pdfs)
        else:
            out = eko.run_dglap(theory, ocard)

        return out

    def run_external(self, theory, ocard, pdf):

        if self.external == "LHA":

            if theory["FNS"] == "FFNS":
                # Reference configuration
                self.skip_pdfs.extend([-5, 5, "T24"])

            from .external import (  # pylint:disable=import-error,import-outside-toplevel
                LHA_utils,
            )

            # here pdf and skip_pdfs is not needed
            return LHA_utils.compute_LHA_data(
                theory,
                ocard,
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )
        elif self.external == "LHAPDF":
            from .external import (  # pylint:disable=import-error,import-outside-toplevel
                lhapdf_utils,
            )

            # here theory card is not needed
            return lhapdf_utils.compute_LHAPDF_data(
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

    def log(self, theory, ocard, pdf, me, ext):

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
