import argparse
import copy
import pathlib
from math import inf, nan

import numpy as np
import pandas as pd
import yaml
from banana import toy

import eko
from eko import basis_rotation as br
from eko.interpolation import lambertgrid
from eko.io import runcards
from eko.io.types import ReferenceRunning
from eko.runner.managed import solve
from ekobox import apply
from ekomark.benchmark.external.LHA_utils import here as there

_sqrt2 = float(np.sqrt(2))

# VFNS theory settings
_t_vfns = dict(
    order=[3, 0],
    couplings=dict(
        alphas=0.35,
        alphaem=0.007496,
        scale=_sqrt2,
        num_flavs_ref=3,
        max_num_flavs=6,
    ),
    heavy=dict(
        num_flavs_init=3,
        num_flavs_max_pdf=6,
        intrinsic_flavors=[],
        masses=[ReferenceRunning([mq, nan]) for mq in (_sqrt2, 4.5, 175.0)],
        masses_scheme="POLE",
        matching_ratios=[1.0, 1.0, 1.0],
    ),
    xif=1.0,
    n3lo_ad_variation=(0, 0, 0, 0),
)
t_vfns = runcards.TheoryCard.from_dict(_t_vfns)

# FFNS theory settings
_t_ffns = copy.deepcopy(_t_vfns)
_t_ffns["couplings"]["num_flavs_ref"] = 4
_t_ffns["heavy"]["num_flavs_init"] = 4
_t_ffns["heavy"]["masses"] = [
    ReferenceRunning([0, nan]),
    ReferenceRunning([inf, nan]),
    ReferenceRunning([inf, nan]),
]
t_ffns = runcards.TheoryCard.from_dict(_t_ffns)

# VFNS operator settings
_o_vfns = dict(
    mu0=_sqrt2,
    mugrid=[(100.0, 5)],
    xgrid=lambertgrid(60).tolist(),
    configs=dict(
        evolution_method="iterate-exact",
        ev_op_max_order=[10, 0],
        ev_op_iterations=30,
        interpolation_polynomial_degree=4,
        interpolation_is_log=True,
        scvar_method="exponentiated",
        inversion_method=None,
        n_integration_cores=1,
        polarized=False,
        time_like=False,
    ),
    debug=dict(
        skip_singlet=False,
        skip_non_singlet=False,
    ),
)
o_vfns = runcards.OperatorCard.from_dict(_o_vfns)

# FFNS operator settings
_o_ffns = copy.deepcopy(_o_vfns)
_o_ffns["mugrid"] = [(100.0, 4)]
o_ffns = runcards.OperatorCard.from_dict(_o_ffns)

# setup flavor rotations
labels = ["u_v", "d_v", "L_m", "L_p", "s_v", "s_p", "c_p", "g"]
rotate_to_LHA = np.zeros((len(labels), 14))
# u_v = u - ubar
rotate_to_LHA[0][br.flavor_basis_pids.index(-2)] = -1
rotate_to_LHA[0][br.flavor_basis_pids.index(2)] = 1
# d_v = d - dbar
rotate_to_LHA[1][br.flavor_basis_pids.index(-1)] = -1
rotate_to_LHA[1][br.flavor_basis_pids.index(1)] = 1
# L_- = dbar - ubar
rotate_to_LHA[2][br.flavor_basis_pids.index(-1)] = 1
rotate_to_LHA[2][br.flavor_basis_pids.index(-2)] = -1
# 2L_+ = 2dbar + 2ubar
rotate_to_LHA[3][br.flavor_basis_pids.index(-1)] = 2
rotate_to_LHA[3][br.flavor_basis_pids.index(-2)] = 2
# s_v = s - sbar
rotate_to_LHA[4][br.flavor_basis_pids.index(-3)] = -1
rotate_to_LHA[4][br.flavor_basis_pids.index(3)] = 1
# s_+ = s + sbar
rotate_to_LHA[5][br.flavor_basis_pids.index(-3)] = 1
rotate_to_LHA[5][br.flavor_basis_pids.index(3)] = 1
# c_+ = c + cbar
rotate_to_LHA[6][br.flavor_basis_pids.index(-4)] = 1
rotate_to_LHA[6][br.flavor_basis_pids.index(4)] = 1
# g = g
rotate_to_LHA[7][br.flavor_basis_pids.index(21)] = 1

# setup x rotation
xgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])

# eko path
p = pathlib.Path("FFNS.tar")

# reference values
with open(there / "LHA.yaml", encoding="utf-8") as o:
    ref_data = yaml.safe_load(o)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", help="Rerun eko", action="store_true")
    args = parser.parse_args()

    # recompute?
    if args.rerun:
        print("Rerunning eko ...")
        p.unlink(True)
        solve(t_ffns, o_ffns, p)

    # apply PDF
    out = {}
    with eko.EKO.read(p) as eko_:
        pdf = apply.apply_pdf_flavor(
            eko_, toy.mkPDF("ToyLH", 0), xgrid, rotate_to_LHA, labels
        )
        for lab, f in pdf[(10000.0, 4)]["pdfs"].items():
            out[lab] = xgrid * f

    # display result
    pd.set_option("display.float_format", "{:.4e}".format)
    me = pd.DataFrame(out)
    print("EKO")
    print(me)

    # load reference
    ref = pd.DataFrame(ref_data["table14"]["part1"])
    print("rel. distance to reference")
    print((me - ref) / ref)
