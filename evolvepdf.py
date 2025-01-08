import pathlib
import eko
from banana import toy

pdf = toy.mkPDF("",0)

from ekobox.apply import apply_pdf

with eko.EKO.read("./benchmarks/data/ekov014.tar") as evolution_operator:           # directory of the EKO object
    evolved_pdfs, _integration_errors = apply_pdf(evolution_operator, pdf)

print(evolved_pdfs.keys())      # can change this to check other things of course, e.g. values using evolved_pdfs[(...)]