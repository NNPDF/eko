# -*- coding: utf-8 -*-
"""
Abstract layer for running the benchmarks
"""
import functools
import logging
import os
import sys

import pandas as pd
from banana.benchmark.runner import BenchmarkRunner
from banana.data import dfdict

import eko
from eko import basis_rotation as br

from .. import pdfname
from ..banana_cfg import banana_cfg
from ..data import db, operators


class Runner(BenchmarkRunner):
    """
    EKO specialization of the banana runner.
    """

    banana_cfg = banana_cfg
    db_base_cls = db.Base
    rotate_to_evolution_basis = False
    sandbox = False
    plot_operator = False

    @staticmethod
    def load_ocards(session, ocard_updates):
        return operators.load(session, ocard_updates)

    @staticmethod
    def skip_pdfs(_theory):
        return []

    def run_me(self, theory, ocard, _pdf):
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
            ops_id = f"o{ocard['hash'][:6]}_t{theory['hash'][:6]}"
            path = f"{banana_cfg['database_path'].parents[0]}/{ops_id}.yaml"

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
                # load
                print(f"Using cached eko data: {os.path.relpath(path,os.getcwd())}")
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
                save_operators_to_pdf(
                    output_path, theory, ocard, out, self.skip_pdfs(theory)
                )
        else:
            out = eko.run_dglap(theory, ocard)

        return out

    def run_external(self, theory, ocard, pdf):
        # pylint:disable=import-error,import-outside-toplevel
        if self.external.lower() == "lha":
            from .external import LHA_utils

            # here pdf and skip_pdfs is not needed
            return LHA_utils.compute_LHA_data(
                theory,
                ocard,
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )
        elif self.external.lower() == "lhapdf":
            from .external import lhapdf_utils

            # here theory card is not needed
            return lhapdf_utils.compute_LHAPDF_data(
                ocard,
                pdf,
                self.skip_pdfs(theory),
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )
        elif self.external.lower() == "pegasus":
            from .external import pegasus_utils

            return pegasus_utils.compute_pegasus_data(
                theory,
                ocard,
                self.skip_pdfs(theory),
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )

        elif self.external.lower() == "apfel":
            from .external import apfel_utils

            return apfel_utils.compute_apfel_data(
                theory,
                ocard,
                pdf,
                self.skip_pdfs(theory),
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )

        raise NotImplementedError(
            f"Benchmark against {self.external} is not implemented!"
        )

    def log(self, theory, _, pdf, me, ext):
        # return a proper log table
        log_tabs = {}
        xgrid = ext["target_xgrid"]
        q2s = list(ext["values"].keys())

        # LHA NNLO VFNS needs a special treatment
        # Valence contains only u and d
        rotate_to_evolution = None
        if self.rotate_to_evolution_basis:
            rotate_to_evolution = br.rotate_flavor_to_evolution.copy()
            if (
                self.external == "LHA"
                and theory["PTO"] == 2
                and theory["FNS"] == "ZM-VFNS"
            ):
                rotate_to_evolution[3, :] = [0, 0, 0, 0, 0, -1, -1, 0, 1, 1, 0, 0, 0, 0]

        pdf_grid = me.apply_pdf_flavor(
            pdf,
            xgrid,
            flavor_rotation=rotate_to_evolution,
        )
        for q2 in q2s:

            log_tab = dfdict.DFdict()
            ref_pdfs = ext["values"][q2]
            res = pdf_grid[q2]
            my_pdfs = res["pdfs"]
            my_pdf_errs = res["errors"]

            for key in my_pdfs:

                if key in self.skip_pdfs(theory):
                    continue

                # build table
                tab = {}
                tab["x"] = xgrid
                tab["Q2"] = q2
                tab["eko"] = f = xgrid * my_pdfs[key]
                tab["eko_error"] = xgrid * my_pdf_errs[key]
                tab[self.external] = r = ref_pdfs[key]
                tab["percent_error"] = (f - r) / r * 100

                log_tab[pdfname(key)] = pd.DataFrame(tab)
            log_tabs[q2] = log_tab

        return functools.reduce(lambda dfd1, dfd2: dfd1.merge(dfd2), log_tabs.values())
