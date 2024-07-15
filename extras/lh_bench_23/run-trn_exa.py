import logging
import pathlib
import sys

import pandas as pd
from banana import toy
from cfg import (
    ffns_labels,
    ffns_operator,
    ffns_rotate_to_LHA,
    ffns_theory,
    n3lo_theory,
    table_dir,
    xgrid,
)

import eko
from eko.io.types import EvolutionMethod
from eko.runner.managed import solve
from ekobox import apply

stdout_log = logging.StreamHandler(sys.stdout)
logger = logging.getLogger("eko")
logger.handlers = []
logger.setLevel(logging.INFO)
logger.addHandler(stdout_log)


def compute(op_card, th_card):
    rot = ffns_rotate_to_LHA
    lab = ffns_labels

    method = op_card.configs.evolution_method.value
    pto = th_card.order[0] - 1
    path = pathlib.Path(f"ekos/FFNS-{pto}_{method}.tar")
    path.unlink(missing_ok=True)

    solve(th_card, op_card, path)

    # apply PDF
    out = {}
    with eko.EKO.read(path) as eko_:
        pdf = apply.apply_pdf_flavor(eko_, toy.mkPDF("ToyLH", 0), xgrid, rot, lab)
        for lab, f in list(pdf.values())[0]["pdfs"].items():
            out[lab] = xgrid * f

    # display result
    pd.set_option("display.float_format", "{:.4e}".format)
    me = pd.DataFrame(out)
    print("EKO")
    print(me)
    # dump to file
    table_dir.mkdir(exist_ok=True)
    me.to_csv(f"{table_dir}/table_FFNS-{pto}_{method}.csv")


if __name__ == "__main__":
    # loop on ev methods
    for ev_method in [EvolutionMethod.TRUNCATED, EvolutionMethod.ITERATE_EXACT]:
        op_card = ffns_operator(ev_method=ev_method.value)
        # loop on pto
        for pto in [1, 2, 3]:
            if pto == 3:
                th_card = n3lo_theory(
                    ad_variation=(0, 0, 0, 0, 0, 0, 0),
                    is_ffns=True,
                    use_fhmruvv=True,
                    xif=1.0,
                )
            else:
                th_card = ffns_theory(xif=1.0, pto=pto)
            compute(op_card, th_card)
