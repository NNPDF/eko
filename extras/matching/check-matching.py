import copy
import logging
import pathlib
import sys

import numpy as np
from banana import toy

import eko
from eko.interpolation import lambertgrid
from ekomark.apply import apply_pdf

t = {
    "ID": 0,
    "HQ": "POLE",
    "ModEv": "EXA",
    "Qmt": 173.07,
    "alphas": 0.35,
    "ktThr": 1.0,
    "PTO": 0,
    "IC": 0,
    "NfFF": 3,
    "Qref": 1.4142135623730951,
    "fact_to_ren_scale_ratio": 1.0,
    "mb": 4.5,
    "CKM": "0.97428 0.22530 0.003470 0.22520 0.97345 0.041000 0.00862 0.04030 0.999152",
    "IB": 0,
    "Q0": 1.4142135623730951,
    "nf0": 3,
    "SIN2TW": 0.23126,
    "global_nx": 0,
    "mc": 1.4142135623730951,
    "Comments": "LO baseline for small-x res",
    "MP": 0.938,
    "QED": 0,
    "SxOrd": "LL",
    "kDISbThr": 1.0,
    "mt": 175.0,
    "DAMP": 0,
    "MW": 80.398,
    "SxRes": 0,
    "TMC": 0,
    "kDIScThr": 1.0,
    "alphaem_running": False,
    "EScaleVar": 1,
    "MZ": 91.1876,
    "nfref": 3,
    "XIF": 1.0,
    "kDIStThr": 1.0,
    "FNS": "ZM-VFNS",
    "MaxNfAs": 6,
    "Qmb": 4.5,
    "XIR": 1.0,
    "kbThr": 1.0,
    "GF": 1.1663787e-05,
    "MaxNfPdf": 6,
    "Qmc": 2.0,
    "alphaqed": 0.007496,
    "kcThr": 1.0,
}


o = {
    "interpolation_xgrid": lambertgrid(60),
    "inputgrid": None,
    "targetgrid": None,
    "inputpids": None,
    "targetpids": None,
    "interpolation_polynomial_degree": 4,
    "debug_skip_non_singlet": False,
    "ev_op_max_order": 10,
    "Q2grid": np.geomspace(4.0, 100.0, 3.0),
    "mtime": None,
    "interpolation_is_log": "1",
    "xgrid": None,
    "debug_skip_singlet": False,
    "ev_op_iterations": 10,
    "backward_inversion": "expanded",
    "n_integration_cores": -1,
}


th_updates = {
    0: {"kbThr": 1.0, "PTO": 0},
    # 1: {"kbThr": 0.5, "PTO": 0},
    # 2: {"kbThr": 2.0, "PTO": 0},
    # 3: {"kbThr": 1.0, "PTO": 1},
    # 4: {"kbThr": 0.5, "PTO": 1},
    # 5: {"kbThr": 2.0, "PTO": 1},
    # 6: {"kbThr": 1.0, "PTO": 2},
    # 7: {"kbThr": 0.5, "PTO": 2},
    # 8: {"kbThr": 2.0, "PTO": 2},
}


def compute():
    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko").handlers = []
    logging.getLogger("eko").addHandler(logStdout)
    logging.getLogger("eko").setLevel(logging.INFO)
    for id, upd in th_updates.items():
        tt = copy.deepcopy(t)
        tt.update(upd)
        p = pathlib.Path(f"./eko_{id}.tar")
        eko.runner.solve(tt, o, p)


def collect_data():
    data = {}
    pdf = toy.mkPDF("", 0)
    for id in th_updates.keys():
        with eko.EKO.open(f"./eko_{id}.tar") as evolution_operator:
            data[id] = apply_pdf(evolution_operator, pdf)
    return data


compute()
