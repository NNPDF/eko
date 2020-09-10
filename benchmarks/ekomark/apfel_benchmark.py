# -*- coding: utf-8 -*-
"""
    Benchmark EKO to APFEL
"""
import yaml
import numpy as np

import eko
from eko import basis_rotation as br

from .toyLH import mkPDF
from .apfel_utils import load_apfel


# generate input pdfs
LHA_init_pdfs = br.generate_input_from_lhapdf(mkPDF("", ""), 2)

class ApfelBenchmark:
    def __init__(self, path):
        with open(path,"r") as o:
            self.cfg = yaml.safe_load(o)
    def run(self):
        output_grid = [.1,.5]
        # # compute our result
        # eko_res = eko.run_dglap(self.cfg)
        # eko_pdf = eko_res.apply_pdf(LHA_init_pdfs,output_grid)
        # print(eko_pdf)
        # import pdb; pdb.set_trace()
        # br.rotate_output(eko_pdf[self.cfg["Q2grid"][0]]["pdfs"])
        # # compute APFEL reference
        # apfel = load_apfel(self.cfg)
        # apfel.EvolveAPFEL(self.cfg["Q0"],np.sqrt(self.cfg["Q2grid"][0]))
        # for x in output_grid:
        #     print(apfel.xPDF(0, x)/x)
        out = eko.output.Output.load_yaml_from_file("assets/fast-ops.yaml")
        print(out)
        eko_pdf = out.apply_pdf(LHA_init_pdfs,output_grid)
        print(eko_pdf)
        import pdb; pdb.set_trace()
        rot = br.rotate_output(list(eko_pdf.values())[0]["pdfs"])
