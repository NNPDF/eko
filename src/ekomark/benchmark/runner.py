"""Abstract layer for running the benchmarks."""

import functools
import logging
import os
import sys

import numpy as np
import pandas as pd
from banana import cfg as banana_cfg
from banana.benchmark.runner import BenchmarkRunner
from banana.data import dfdict

import eko
from eko import EKO
from eko import basis_rotation as br
from eko.io.runcards import Legacy
from ekobox import apply

from .. import pdfname
from ..data import db, operators


class Runner(BenchmarkRunner):
    """EKO specialization of the banana runner."""

    db_base_cls = db.Base
    rotate_to_evolution_basis = False
    sandbox = False
    plot_operator = False
    log_to_stdout = True

    def __init__(self):
        self.banana_cfg = banana_cfg.cfg

    @staticmethod
    def load_ocards(session, ocard_updates):
        """Load operator cards.

        Parameters
        ----------
        session : sqlalchemy.session.Session
            DB ORM session
        updates : dict
            modifiers

        Returns
        -------
        cards : list(dict)
            list of records
        """
        return operators.load(session, ocard_updates)

    @staticmethod
    def skip_pdfs(_theory):
        """Specify PDFs to skip."""
        return []

    def run_me(self, raw_theory, raw_ocard, _pdf):
        """Run eko.

        Parameters
        ----------
        raw_theory : dict
            raw theory card
        raw_ocard : dict
            raw operator card
        pdf : lhapdf_type
            pdf

        Returns
        -------
            out :  dict
                DGLAP result
        """
        # activate logging
        if self.log_to_stdout:
            logStdout = logging.StreamHandler(sys.stdout)
            logStdout.setLevel(logging.INFO)
            logStdout.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger("eko").handlers = []
            logging.getLogger("eko").addHandler(logStdout)
            logging.getLogger("eko").setLevel(logging.INFO)

        ops_id = f"o{raw_ocard['hash'][:6]}_t{raw_theory['hash'][:6]}"
        root = banana_cfg.cfg["paths"]["database"].parents[0]
        path = root / f"{ops_id}.tar"

        # convert cards
        conv = Legacy(raw_theory, raw_ocard)
        theory = conv.new_theory
        ocard = conv.new_operator

        # if sandbox check for cache, dump eko to yaml
        # and plot the operators
        if self.sandbox:
            rerun = True

            if os.path.exists(path):
                rerun = False
                ask = input("Use cached output? [Y/n]")
                if ask.lower() in ["n", "no"]:
                    rerun = True

            if rerun:
                path.unlink(missing_ok=True)
                eko.solve(theory, ocard, path)
                print(f"Operator written to {path}")
            else:
                # load
                print(f"Using cached eko data: {os.path.relpath(path, os.getcwd())}")

            if self.plot_operator:
                from ekomark.plots import (  # pylint:disable=import-error,import-outside-toplevel
                    save_operators_to_pdf,
                )

                output_path = f"{root}/{self.external}_bench"
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                # rotating to evolution basis if requested
                with EKO.edit(path) as out_copy:
                    save_operators_to_pdf(
                        output_path,
                        raw_theory,
                        raw_ocard,
                        out_copy,
                        self.skip_pdfs(raw_theory),
                        self.rotate_to_evolution_basis,
                        raw_theory["QED"] > 0,
                    )
        else:
            # else we always rerun
            path.unlink(missing_ok=True)
            eko.solve(theory, ocard, path)
        return path

    def run_external(self, theory, ocard, pdf):
        """Run external library.

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
        # pylint:disable=import-error,import-outside-toplevel
        if self.external.lower() == "void":
            xgrid = ocard["interpolation_xgrid"]
            mugrid = ocard["mugrid"]
            labels = br.flavor_basis_pids
            if self.rotate_to_evolution_basis:
                labels = br.unified_evol_basis if theory["QED"] > 0 else br.evol_basis

            void = {
                "target_xgrid": xgrid,
                "values": {
                    mu**2: {pid: [0.0] * len(xgrid) for pid in labels} for mu in mugrid
                },
            }
            return void
        if self.external.lower() == "lha":
            from .external import LHA_utils

            # here pdf and skip_pdfs is not needed
            return LHA_utils.compute_LHA_data(
                theory,
                ocard,
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )
        if self.external.lower() == "lhapdf":
            from .external import lhapdf_utils

            # here theory card is not needed
            return lhapdf_utils.compute_LHAPDF_data(
                theory,
                ocard,
                pdf,
                self.skip_pdfs(theory),
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )
        if self.external.lower() == "pegasus":
            from .external import pegasus_utils

            return pegasus_utils.compute_pegasus_data(
                theory,
                ocard,
                self.skip_pdfs(theory),
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )

        if self.external.lower() == "apfel":
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
        """Return a proper log table.

        Parameters
        ----------
            theory : dict
                theory card
            pdf : lhapdf_type
                pdf
            me : eko.output.Output
                eko output object containing all informations
        """
        # return a proper log table
        log_tabs = {}
        xgrid = ext["target_xgrid"]

        # LHA NNLO VFNS needs a special treatment
        # Valence contains only u and d
        rotate_to_evolution = None
        labels = br.flavor_basis_pids
        qed = theory["QED"] > 0
        if self.rotate_to_evolution_basis:
            if not qed:
                rotate_to_evolution = br.rotate_flavor_to_evolution.copy()
                labels = br.evol_basis
            else:
                rotate_to_evolution = br.rotate_flavor_to_unified_evolution.copy()
                labels = br.unified_evol_basis
            if (
                self.external == "LHA"
                and theory["PTO"] == 2
                and theory["FNS"] == "ZM-VFNS"
            ):
                rotate_to_evolution[3, :] = [0, 0, 0, 0, 0, -1, -1, 0, 1, 1, 0, 0, 0, 0]

        with EKO.read(me) as eko:
            pdf_grid, errors = apply.apply_pdf_flavor(
                eko,
                pdf,
                labels,
                xgrid,
                flavor_rotation=rotate_to_evolution,
            )
        for q2, ref_pdfs in ext["values"].items():
            log_tab = dfdict.DFdict()
            # find the first hit, regardless of the flavor, since others can not deal with them appearing in parallel
            eps = [ep for ep in pdf_grid.keys() if np.isclose(ep[0], q2)]
            if len(eps) != 1:
                if len(eps) == 0:
                    raise KeyError(f"PDF at Q2={q2} not found")
                raise KeyError(f"More than one evolution point found for Q2={q2}")
            my_pdfs = pdf_grid[eps[0]]
            my_pdf_errs = errors[eps[0]]

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
