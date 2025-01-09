import pathlib

TEST = pathlib.Path(__file__).parents[2] / "data"

import pathlib
import eko
from banana import toy

pdf = toy.mkPDF("",0)

from ekobox.apply import apply_pdf

#print(evolved_pdfs.keys())
def test_read_legacy():
    for name in ["v0.13.tar"]:
        with eko.EKO.read(TEST/name) as evolution_operator:           # directory of the EKO object
            evolved_pdfs, _integration_errors = apply_pdf(evolution_operator, pdf)
            #import pdb; pdb.set_trace()
            #print('test')
            assert isinstance(evolution_operator.theory_card, eko.io.runcards.TheoryCard)