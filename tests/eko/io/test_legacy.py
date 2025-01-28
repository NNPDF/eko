import pathlib
from math import nan
TEST = pathlib.Path(__file__).parents[2] / "data"   # directory of the EKO object

import pathlib
import eko
from banana import toy
import yaml
import numpy as np
from eko.interpolation import make_grid
from ekobox.evol_pdf import evolve_pdfs
import lhapdf

pdf = toy.mkPDF("",0)

from ekobox.apply import apply_pdf

x_grid = [1E-7,1E-6,1E-5,1E-4,1E-3,1E-2,0.1,0.3,0.5,0.7,0.9]
Q2 = 10000

def test_read_legacy():
    for name in ["v0.14.tar"]:
    
        with eko.EKO.read(TEST/name) as evolution_operator:
            #import pdb; pdb.set_trace()
            assert isinstance(evolution_operator.theory_card, eko.io.runcards.TheoryCard)   # Check that theory card is read as theory card
            assert isinstance(evolution_operator.operator_card, eko.io.runcards.OperatorCard)   # Check that operator card is read as operator card
        
            evolved_pdfs, _integration_errors = apply_pdf(evolution_operator, pdf, x_grid)

        
        
        pdf_test = evolved_pdfs[10000,4][21]    # evolved gluon PDF at 10000 GeV^2

        # Import the values of the LHA benchmark tables. This is not very nice yet, but should work
        lha_path = pathlib.Path(__file__).parents[4] / 'eko/src/ekomark/benchmark/external/LHA.yaml'
        with open(lha_path,'r') as file:
            lha_benchmark = yaml.safe_load(file)

        xpdf_benchmark = lha_benchmark["table2"]["part2"]["g"]        

        pdf_benchmark = []      # gluon PDF at 10000 GeV^2 from the LHA benchmark tables
        """
        --> have to divide by x values to compare or not
        Was the LHAPDF output xf(x,q2) or f(x,q2) again?
        """
        for j in range(len(xpdf_benchmark)):
            pdf_benchmark.append(xpdf_benchmark[j]/x_grid[j])

        print(pdf_test, pdf_benchmark)
        
        # Test that the PDF values are the same 
        for i in range(len(pdf_test)):
            if pdf_test[i] != pdf_benchmark[i]:
                print(name, False)
            else:
                print(name, True)
       
    return evolved_pdfs

test_read_legacy()