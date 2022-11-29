import copy
import os

import numpy as np
import numba as nb
from ekopolarised import anomalous_dimensions as ad
from ekopolarised.anomalous_dimensions.__init__ import gamma_singlet
from ekopolarised import basis_rotation as br
from ekopolarised import interpolation, mellin
from ekopolarised.couplings import Couplings
from ekopolarised.evolution_operator import Operator, quad_ker, select_singlet_element
from ekopolarised.evolution_operator.grid import OperatorGrid
from ekopolarised.interpolation import InterpolatorDispatcher, XGrid
from ekopolarised.kernels import non_singlet as ns
from ekopolarised.kernels import singlet as s
from ekopolarised.thresholds import ThresholdsAtlas
from ekopolarised import beta
from ekopolarised import interpolation
from ekopolarised import runner 

#declare parameter values
order= (1, 0)
n = complex(1.0, 1.0)
nf= 4
p = False
a1= 2
a0 = 1

#modes used
mode0=br.non_singlet_pids_map["ns+"]
mode1=0


#using anomalous dimensions singlet, make that matrix 
gamma_s= gamma_singlet(order, n, nf, p)
#make the evolution kernel with starting and ending energy 


j00= np.log(a1 / a0) / beta.beta_qcd((2, 0), nf)
lo_ex = ad.exp_singlet(gamma_s[0] * j00)

print (lo_ex)
ker_s = lo_ex
print(np.shape(ker_s))
k = 0 if mode0 == 100 else 1 
l = 0 if mode1 == 100 else 1
print(k)
print(l)
select_el= ker_s[k]
print (select_el)

spec = [
    ("is_singlet", nb.boolean),
    ("is_log", nb.boolean),
    ("logx", nb.float64),
    ("u", nb.float64),
]

#######################
theory_card = {
    "alphas": 0.35,
    "alphaqed": 0.007496,
    "PTO": 0,
    "QED": 0,
    "fact_to_ren_scale_ratio": 1.0,
    "Qref": np.sqrt(2),
    "nfref": 4,
    "Q0": np.sqrt(2),
    "nf0": 4,
    "p": True,
    "FNS": "FFNS",
    "NfFF": 3,
    "ModEv": "EXA",
    "IC": 0,
    "IB": 0,
    "mc": 1.0,
    "mb": 4.75,
    "mt": 173.0,
    "kcThr": 0,
    "kbThr": np.inf,
    "ktThr": np.inf,
    "MaxNfPdf": 6,
    "MaxNfAs": 6,
    "HQ": "MSBAR",
    "Qmc": 1.0,
    "Qmb": 4.75,
    "Qmt": 173.0,
    "ModSV": None,
}


operators_card = {
    "Q2grid": [10, 100],
    "Q0": np.sqrt(2),
    "configs": {
        "interpolation_polynomial_degree": 1,
        "interpolation_is_log": True,
        "ev_op_max_order": (2, 0),
        "ev_op_iterations": 1,
        "backward_inversion": "exact",
        "n_integration_cores": 1,
    },
    "rotations": {
        "xgrid": [0.01, 0.1, 1.0],
        "pids": np.array(br.flavor_basis_pids),
        "inputgrid": None,
        "targetgrid": None,
        "inputpids": None,
        "targetpids": None,
    },
    "debug": {"skip_singlet": True, "skip_non_singlet": True},
}


def test_raw(theory_card, operators_card):
    """we don't check the content here, but only the shape"""
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operators_card)
    r = runner.Runner(tc, oc)
    o = r.get_output()
    print (r)
    check_shapes(o, o.xgrid, o.xgrid, tc, oc)


def test_targetgrid():
    # change targetgrid
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operators_card)
    igrid = [0.1, 1.0]
    oc["rotations"]["inputgrid"] = igrid
    tgrid = [0.1, 1.0]
    oc["rotations"]["targetgrid"] = tgrid
    r = runner.Runner(tc, oc)
    o = r.get_output()
    print(o)
    check_shapes(
        o, interpolation.XGrid(tgrid), interpolation.XGrid(igrid), tc, oc
    )
    # check actual value
    np.testing.assert_allclose(o.rotations.targetgrid.raw, tgrid)


def test_rotate_pids():
    # change pids
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operators_card)
    oc["rotations"]["targetpids"] = np.eye(14) + 0.1 * np.random.rand(14, 14)
    oc["rotations"]["inputpids"] = np.eye(14) + 0.1 * np.random.rand(14, 14)
    r = runner.Runner(tc, oc)
    o = r.get_output()
    check_shapes(o, o.xgrid, o.xgrid, tc, oc)
    # check actual values
    assert (o.rotations.targetpids == [0] * 14).all()
    assert (o.rotations.inputpids == [0] * 14).all()


def check_shapes(o, txs, ixs, theory_card, operators_card):
    tpids = len(o.rotations.targetpids)
    ipids = len(o.rotations.inputpids)
    op_shape = (tpids, len(txs), ipids, len(ixs))

    # check output = input
    np.testing.assert_allclose(o.xgrid.raw, operators_card["rotations"]["xgrid"])
    np.testing.assert_allclose(o.rotations.targetgrid.raw, txs.raw)
    np.testing.assert_allclose(o.rotations.inputgrid.raw, ixs.raw)
    for k in ["interpolation_polynomial_degree", "interpolation_is_log"]:
        assert getattr(o.configs, k) == operators_card["configs"][k]
    np.testing.assert_allclose(o.Q02, theory_card["Q0"] ** 2)
    # check available operators
    assert len(o.Q2grid) == len(operators_card["Q2grid"])
    assert list(o.Q2grid) == operators_card["Q2grid"]
    for _, ops in o.items():
        assert ops.operator.shape == op_shape
        assert ops.error.shape == op_shape


def test_vfns():
    # change targetpids
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operators_card)
    tc["kcThr"] = 1.0
    tc["kbThr"] = 1.0
    tc["order"] = (3, 0)
    oc["debug"]["skip_non_singlet"] = False
    # tc,oc = compatibility.update(theory_card,operators_card)
    r = runner.Runner(tc, oc)
    o = r.get_output()
    check_shapes(o, o.xgrid, o.xgrid, tc, oc)


test_targetgrid()
test_rotate_pids()



from ekopolarised import output, runner, version

__version__ = version.__version__

def run_dglap(theory_card, operators_card):
    r = runner.Runner(theory_card, operators_card)
    output = r.get_output()
    return output

run_dglap(theory_card, operators_card)


