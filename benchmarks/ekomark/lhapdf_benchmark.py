# -*- coding: utf-8 -*-
"""
Benchmark EKO to APFEL
"""
import numpy as np

import lhapdf

import eko
from eko import basis_rotation as br

from .runner import Runner


class LHAPDFBenchmark(Runner):
    """
    Benchmark EKO to LHAPDF

    Parameters
    ----------
        theory_path : string or pathlib.Path
            path to theory card
        operators_path : string or pathlib.Path
            path to operators card
        assets_dir : string
            output directory
    """

    def __init__(self, theory_path, operators_path, pdf, assets_dir):
        super().__init__(theory_path, operators_path, assets_dir)
        self.target_xgrid = eko.interpolation.make_grid(
            *self.operators["interpolation_xgrid"][1:]
        )
        self.src_pdf = pdf
        self.skip_pdfs = [22, -6, -5, 5, 6]
        self.rotate_to_evolution_basis = True

    def ref(self):
        return {
            "target_xgrid": self.target_xgrid,
            "values": {self.operators["Q2grid"][0]: self.ref_values()},
            "src_pdf": self.src_pdf,
            "rotate_to_evolution_basis": self.rotate_to_evolution_basis,
            "skip_pdfs": self.skip_pdfs,
        }

    def ref_values(self):
        """
        Run LHAPDF
        """
        pdf = lhapdf.mkPDF(self.src_pdf, 0)
        apf_tabs = {}
        for pid in br.flavor_basis_pids:
            # skip?
            if pid in self.skip_pdfs:
                continue
            # collect APFEL
            apf = []
            for x in self.target_xgrid:
                xf = pdf.xfxQ2(pid, x, self.operators["Q2grid"][0])
                apf.append(xf)
            apf_tabs[pid] = np.array(apf)
        # rotate if needed
        if self.rotate_to_evolution_basis:
            pdfs = np.array(
                [
                    apf_tabs[pid]
                    if pid in apf_tabs
                    else np.zeros(len(self.target_xgrid))
                    for pid in br.flavor_basis_pids
                ]
            )
            evol_pdf = br.rotate_flavor_to_evolution @ pdfs
            apf_tabs = dict(zip(br.evol_basis, evol_pdf))
        return apf_tabs
