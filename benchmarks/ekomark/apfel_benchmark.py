# -*- coding: utf-8 -*-
"""
Benchmark EKO to APFEL
"""
import time
import numpy as np

import eko

from .apfel_utils import load_apfel
from .runner import Runner


class ApfelBenchmark(Runner):
    """
    Benchmark EKO to APFEL

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
        super().__init__(theory_path, operators_path, assets_dir)
        self.target_xgrid = eko.interpolation.make_grid(
            *self.operators["interpolation_xgrid"][1:]
        )

    def ref(self):
        return {
            "target_xgrid": self.target_xgrid,
            "values": {self.operators["Q2grid"][0]: self.ref_values()},
            "src_pdf": "ToyLH",
            "is_flavor_basis": True,
            "skip_pdfs": [],
        }

    def ref_values(self):
        """
        Run APFEL
        """
        # compute APFEL reference
        apf_start = time.perf_counter()
        apfel = load_apfel(self.theory, self.operators, "ToyLH")
        print("Loading APFEL took %f s" % (time.perf_counter() - apf_start))
        apfel.EvolveAPFEL(self.theory["Q0"], np.sqrt(self.operators["Q2grid"][0]))
        print("Executing APFEL took %f s" % (time.perf_counter() - apf_start))
        apf_tabs = {}
        for pid in [-4, -3, -2, -1, 21, 1, 2, 3, 4]:
            # collect APFEL
            apf = []
            for x in self.target_xgrid:
                apf.append(apfel.xPDF(pid if pid < 21 else 0, x))
            apf_tabs[pid] = np.array(apf)
        return apf_tabs

    def print(self, apf_tabs):
        """
        Print all result

        Parameters
        ----------
            apf_tabs : dict
                comparison result
        """
        # iterate all values
        for q2, dfdict in apf_tabs.items():
            print("-" * 20)
            print(f"Q2 = {q2} GeV^2 ")
            print("-" * 20)
            print(dfdict)
            print("-" * 20)
