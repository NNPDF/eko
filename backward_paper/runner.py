# -*- coding: utf-8 -*-
"""
This script contains a specialization of the Ekomark runner
"""
import functools
import copy
import pandas as pd
import numpy as np

from banana import load_config
from banana.data import dfdict
from banana.benchmark.runner import get_pdf_name

from ekomark.benchmark.runner import Runner
from ekomark import pdfname
from ekomark.benchmark.external.lhapdf_utils import compute_LHAPDF_data

from plots import plot_pdf
from config import pkg_path

import eko


def rotate_to_pm_basis(log, skip=None):
    """
    Rotate to plus minus basis

    Parameters
    ----------
        log: dict
            log table
        skip: str
            skip '+' or '-'
    """
    rot_log = {}
    skip = skip if skip is not None else []
    if "g" in log and "g" not in skip:
        rot_log["g"] = log["g"]
    for pid in eko.evolution_operator.flavors.quark_names:
        if pid not in log:
            continue
        quark = log[pid]
        qbar = log[f"{pid}bar"].copy()

        for key, fact in zip(["+", "-"], [1, -1]):
            if key in skip:
                continue
            rot_log[r"${%s}^{%s}$" % (pid, key)] = copy.deepcopy(quark)
            for column_name in quark.iloc[:, 1:]:
                if "error" in column_name:
                    continue
                rot_log[r"${%s}^{%s}$" % (pid, key)][column_name] += (
                    fact * qbar[column_name]
                )
    return rot_log


class BackwardPaperRunner(Runner):
    """
    Specialization of the Ekomark runner.
    """

    def __init__(self):
        super().__init__()
        self.banana_cfg = load_config(pkg_path)
        self.rotate_to_evolution_basis = False
        self.plot_pdfs = [4, -4]
        self.sandbox = True
        self.fig_name = None
        self.return_to_Q0 = False
        self.ekos = []
        self.intermediate_Q = None

    def run_back_forth(self, theory_updates, ocard_updates, pdfs, use_replicas=True):
        """Compute the sequence q0->q1->q0"""
        self.ekos = []
        for idx, q0 in enumerate(theory_updates[0]["Q0"]):
            temp_theory = copy.deepcopy(theory_updates)
            temp_theory[0]["Q0"] = q0
            super().run(
                temp_theory, [ocard_updates[idx]], pdfs, use_replicas=use_replicas
            )

    def run_me(self, theory, ocard, _pdf):
        # need to run or do an eko product?
        if len(self.ekos) != 2:
            me = super().run_me(theory, ocard, _pdf)
            # need to store the eko here?
            if self.return_to_Q0:
                self.ekos.append(me)
        else:
            self.console.print("Computing product between the previous Ekos")

            # get the operator tensors
            op_list = []
            for eko in self.ekos:
                for operators in eko["Q2grid"].values():
                    op_list.append(operators["operators"])

            # do tensor product
            final_op = np.einsum("ajbk,bkcl -> ajcl", op_list[1], op_list[0])

            # set the final eko values
            me = copy.deepcopy(self.ekos[1])
            self.intermediate_Q = np.round(np.sqrt(self.ekos[1]["q2_ref"]), 2)
            q2_from = theory["Q0"] ** 2
            me["q2_ref"] = q2_from
            me["Q2grid"][q2_from]["operators"] = final_op

            # test?
            # for idx, op in enumerate(final_op):
            #     # skip ph, t, tbar,
            #     if idx in [0,1,13]:
            #         continue
            #     np.testing.assert_allclose(op[:,idx,:], np.eye(final_op.shape[1]), atol=0.01)
        return me

    def run_external(self, theory, ocard, pdf):

        if self.external == "inputpdf":
            # Compare with the initial pdf
            ext = compute_LHAPDF_data(
                ocard,
                pdf,
                skip_pdfs=[22],
                Q2s=[theory["Q0"] ** 2],
                rotate_to_evolution_basis=self.rotate_to_evolution_basis,
            )
        return ext

    def log(self, theory, ocard, pdf, me, ext):
        """Apply PDFs to eko and produce log tables"""
        log_tabs = {}
        xgrid = ocard["interpolation_xgrid"]
        q2s = me["Q2grid"]

        rotate_to_evolution = None
        if self.rotate_to_evolution_basis:
            rotate_to_evolution = eko.basis_rotation.rotate_flavor_to_evolution.copy()

        pdf_name = get_pdf_name(pdf)

        # build table
        tab = {}
        tab["x"] = xgrid

        # Loop over pdfs replicas
        for rep, pdf_set in enumerate(pdf):
            pdf_grid = me.apply_pdf_flavor(
                pdf_set,
                xgrid,
                flavor_rotation=rotate_to_evolution,
            )

            log_tab = dfdict.DFdict()
            ref_pdfs = ext[rep]

            # Loop over pdf ids
            for key in self.plot_pdfs:

                if self.external == "inputpdf":
                    tab[f'{pdf_name}_@_{theory["Q0"]}'] = ref_pdfs["values"][
                        theory["Q0"] ** 2
                    ][key]

                # Loop over q2 grid
                for q2 in q2s:
                    res = pdf_grid[q2]
                    my_pdfs = res["pdfs"]
                    # my_pdf_errs = res["errors"]
                    if self.intermediate_Q:
                        tab[
                            f"EKO_@_{np.round(np.sqrt(q2), 2)}_>_{self.intermediate_Q}_>_{np.round(np.sqrt(q2), 2)}"
                        ] = (xgrid * my_pdfs[key])
                    else:
                        tab[f"EKO_@_{np.round(np.sqrt(q2), 2)}"] = xgrid * my_pdfs[key]

                log_tab[pdfname(key)] = pd.DataFrame(tab)
            log_tabs[rep] = log_tab

        # Merge dfd for different replicas
        new_log = functools.reduce(
            lambda dfd1, dfd2: dfd1.merge(dfd2), log_tabs.values()
        )

        # skip plotting for intermediate step when
        # returning to Q0
        if self.return_to_Q0 and self.intermediate_Q is None:
            return new_log

        # Plot
        plot_pdf(
            rotate_to_pm_basis(new_log, skip="minus")
            if not self.rotate_to_evolution_basis
            else new_log,
            self.fig_name,
            cl=1,
            plot_reldiff=self.return_to_Q0,
        )

        return new_log
