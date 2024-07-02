import copy
import pathlib
from math import inf, nan

import numpy as np

from eko import basis_rotation as br
from eko.interpolation import lambertgrid
from eko.io import runcards
from eko.io.types import ReferenceRunning

here = pathlib.Path(__file__).parent
eko_dir = here / "ekos"
table_dir = here / "tables"


_sqrt2 = float(np.sqrt(2))

# setup x rotation
xgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])

# theory settings
# ---------------
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
    n3lo_ad_variation=(0, 0, 0, 0, 0, 0, 0),
    matching_order=[2, 0],
    use_fhmruvv=False,
)


def vfns_theory(xif=1.0):
    """Generate a VFNS theory card."""
    tt = copy.deepcopy(_t_vfns)
    tt["xif"] = xif
    return runcards.TheoryCard.from_dict(tt)


_t_ffns = copy.deepcopy(_t_vfns)
_t_ffns["couplings"]["num_flavs_ref"] = 4
_t_ffns["heavy"]["num_flavs_init"] = 4
_t_ffns["heavy"]["masses"] = [
    ReferenceRunning([0, nan]),
    ReferenceRunning([inf, nan]),
    ReferenceRunning([inf, nan]),
]


def ffns_theory(xif=1.0, pto=2):
    """Generate a VFNS theory card."""
    tt = copy.deepcopy(_t_ffns)
    tt["xif"] = xif
    tt["order"] = (pto + 1, 0)
    tt["matching_order"] = (pto, 0)
    return runcards.TheoryCard.from_dict(tt)


def n3lo_theory(ad_variation, is_ffns, use_fhmruvv=False, xif=1.0):
    """Generate an N3LO theory card."""
    base = _t_ffns if is_ffns else _t_vfns
    tt = copy.deepcopy(base)
    tt["xif"] = xif
    tt["order"] = [4, 0]
    # here we keep the NNLO matching
    tt["matching_order"] = [2, 0]
    tt["n3lo_ad_variation"] = ad_variation
    tt["use_fhmruvv"] = use_fhmruvv
    return runcards.TheoryCard.from_dict(tt)


# operator settings
# -----------------
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
        n_integration_cores=-2,
        polarized=False,
        time_like=False,
    ),
    debug=dict(
        skip_singlet=False,
        skip_non_singlet=False,
    ),
)
vfns_operator = runcards.OperatorCard.from_dict(_o_vfns)


def ffns_operator(ev_method="iterate-exact"):
    """Generate a FFNS theory card."""
    op = copy.deepcopy(_o_vfns)
    op["mugrid"] = [(100.0, 4)]
    op["configs"]["evolution_method"] = ev_method
    if ev_method == "truncated":
        op["configs"]["ev_op_iterations"] = 1
    return runcards.OperatorCard.from_dict(op)


# flavor rotations
# ----------------

ffns_labels = ["u_v", "d_v", "L_m", "L_p", "s_v", "s_p", "c_p", "g"]
ffns_rotate_to_LHA = np.zeros((len(ffns_labels), 14))
# u_v = u - ubar
ffns_rotate_to_LHA[0][br.flavor_basis_pids.index(-2)] = -1
ffns_rotate_to_LHA[0][br.flavor_basis_pids.index(2)] = 1
# d_v = d - dbar
ffns_rotate_to_LHA[1][br.flavor_basis_pids.index(-1)] = -1
ffns_rotate_to_LHA[1][br.flavor_basis_pids.index(1)] = 1
# L_- = dbar - ubar
ffns_rotate_to_LHA[2][br.flavor_basis_pids.index(-1)] = 1
ffns_rotate_to_LHA[2][br.flavor_basis_pids.index(-2)] = -1
# 2L_+ = 2dbar + 2ubar
ffns_rotate_to_LHA[3][br.flavor_basis_pids.index(-1)] = 2
ffns_rotate_to_LHA[3][br.flavor_basis_pids.index(-2)] = 2
# s_v = s - sbar
ffns_rotate_to_LHA[4][br.flavor_basis_pids.index(-3)] = -1
ffns_rotate_to_LHA[4][br.flavor_basis_pids.index(3)] = 1
# s_+ = s + sbar
ffns_rotate_to_LHA[5][br.flavor_basis_pids.index(-3)] = 1
ffns_rotate_to_LHA[5][br.flavor_basis_pids.index(3)] = 1
# c_+ = c + cbar
ffns_rotate_to_LHA[6][br.flavor_basis_pids.index(-4)] = 1
ffns_rotate_to_LHA[6][br.flavor_basis_pids.index(4)] = 1
# g = g
ffns_rotate_to_LHA[7][br.flavor_basis_pids.index(21)] = 1

vfns_labels = ["u_v", "d_v", "L_m", "L_p", "s_p", "c_p", "b_p", "g"]
vfns_rotate_to_LHA = np.zeros((len(vfns_labels), 14))
# u_v = u - ubar
vfns_rotate_to_LHA[0][br.flavor_basis_pids.index(-2)] = -1
vfns_rotate_to_LHA[0][br.flavor_basis_pids.index(2)] = 1
# d_v = d - dbar
vfns_rotate_to_LHA[1][br.flavor_basis_pids.index(-1)] = -1
vfns_rotate_to_LHA[1][br.flavor_basis_pids.index(1)] = 1
# L_- = dbar - ubar
vfns_rotate_to_LHA[2][br.flavor_basis_pids.index(-1)] = 1
vfns_rotate_to_LHA[2][br.flavor_basis_pids.index(-2)] = -1
# 2L_+ = 2dbar + 2ubar
vfns_rotate_to_LHA[3][br.flavor_basis_pids.index(-1)] = 2
vfns_rotate_to_LHA[3][br.flavor_basis_pids.index(-2)] = 2
# s_+ = s + sbar
vfns_rotate_to_LHA[4][br.flavor_basis_pids.index(-3)] = 1
vfns_rotate_to_LHA[4][br.flavor_basis_pids.index(3)] = 1
# c_+ = c + cbar
vfns_rotate_to_LHA[5][br.flavor_basis_pids.index(-4)] = 1
vfns_rotate_to_LHA[5][br.flavor_basis_pids.index(4)] = 1
# b_+ = b + bbar
vfns_rotate_to_LHA[6][br.flavor_basis_pids.index(-5)] = 1
vfns_rotate_to_LHA[6][br.flavor_basis_pids.index(5)] = 1
# g = g
vfns_rotate_to_LHA[7][br.flavor_basis_pids.index(21)] = 1
