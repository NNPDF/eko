import argparse
import logging
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml
from banana import toy
from cfg import (
    ffns_labels,
    ffns_operator,
    ffns_rotate_to_LHA,
    ffns_theory,
    vfns_labels,
    vfns_operator,
    vfns_rotate_to_LHA,
    vfns_theory,
)

import eko
from eko.runner.managed import solve
from ekobox import apply
from ekomark.benchmark.external.LHA_utils import here as there

_sqrt2 = float(np.sqrt(2))


# setup x rotation
xgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])


# reference values
with open(there / "LHA.yaml", encoding="utf-8") as o:
    ref_data = yaml.safe_load(o)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scheme", help="FFNS or VFNS?")
    parser.add_argument("sv", help="scale variation: up, central, or down")
    parser.add_argument("--rerun", help="Rerun eko", action="store_true")
    parser.add_argument(
        "-v", "--verbose", help="Print eko log to screen", action="store_true"
    )
    args = parser.parse_args()

    # determine xif
    if "central".startswith(args.sv):
        xif = 1.0
        sv = "central"
        part = 1
    elif "up".startswith(args.sv):
        xif = _sqrt2
        sv = "up"
        part = 2
    elif "down".startswith(args.sv):
        xif = 1.0 / _sqrt2
        sv = "down"
        part = 3
    else:
        raise ValueError(
            "sv has to be up, central, or down - or any abbreviation there of"
        )

    # determine scheme
    if args.scheme == "FFNS":
        scheme = "FFNS"
        t = ffns_theory(xif)
        o = ffns_operator
        tab = 14
        lab = ffns_labels
        rot = ffns_rotate_to_LHA
    elif args.scheme == "VFNS":
        scheme = "VFNS"
        t = vfns_theory(xif)
        o = vfns_operator
        tab = 15
        lab = vfns_labels
        rot = vfns_rotate_to_LHA
    else:
        raise ValueError("scheme has to be FFNS or VFNS")

    # eko path
    p = pathlib.Path(f"{scheme}-{sv}.tar")

    # recompute?
    if not p.exists() or args.rerun:
        print("(Re)running eko ...")
        p.unlink(True)
        if args.verbose:
            logStdout = logging.StreamHandler(sys.stdout)
            logStdout.setLevel(logging.INFO)
            logStdout.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger("eko").handlers = []
            logging.getLogger("eko").addHandler(logStdout)
            logging.getLogger("eko").setLevel(logging.INFO)
        solve(t, o, p)

    # apply PDF
    out = {}
    with eko.EKO.read(p) as eko_:
        pdf = apply.apply_pdf_flavor(eko_, toy.mkPDF("ToyLH", 0), xgrid, rot, lab)
        for lab, f in list(pdf.values())[0]["pdfs"].items():
            out[lab] = xgrid * f

    # display result
    pd.set_option("display.float_format", "{:.4e}".format)
    me = pd.DataFrame(out)
    print("EKO")
    print(me)
    # dump to file
    me.to_csv(f"table{tab}-part{part}.csv")

    # load reference
    ref = pd.DataFrame(ref_data[f"table{tab}"][f"part{part}"])
    print()
    print("rel. distance to reference")
    print((me - ref) / ref)
