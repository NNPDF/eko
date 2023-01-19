import copy
import logging
import pathlib
import sys

import matplotlib.pyplot as plt
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
    "Q2grid": np.geomspace(4.0, 100.0, 3),
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
    1: {"kbThr": 0.5, "PTO": 0},
    2: {"kbThr": 2.0, "PTO": 0},
    # 3: {"kbThr": 1.0, "PTO": 1},
    # 4: {"kbThr": 0.5, "PTO": 1},
    # 5: {"kbThr": 2.0, "PTO": 1},
    # 6: {"kbThr": 1.0, "PTO": 2},
    # 7: {"kbThr": 0.5, "PTO": 2},
    # 8: {"kbThr": 2.0, "PTO": 2},
}


def get_theory(id: int) -> dict:
    tt = copy.deepcopy(t)
    tt.update(th_updates[id])
    return tt


def compute():
    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko").handlers = []
    logging.getLogger("eko").addHandler(logStdout)
    logging.getLogger("eko").setLevel(logging.INFO)
    for tid in th_updates.keys():
        tt = get_theory(tid)
        p = pathlib.Path(f"./eko_{tid}.tar")
        eko.runner.solve(tt, o, p)


def collect_data():
    data = {}
    pdf = toy.mkPDF("", 0)
    for id in th_updates.keys():
        with eko.EKO.open(f"./eko_{id}.tar") as evolution_operator:
            data[id] = {
                q2: el["pdfs"] for q2, el in apply_pdf(evolution_operator, pdf).items()
            }
    return data


def log_minor_ticks(min, max, base=10.0, scale=1.0, n=10, majors=None):
    if majors is None:
        lmin = np.ceil(np.log(min / scale) / np.log(base))
        lmax = np.floor(np.log(max / scale) / np.log(base))

        majors = scale * base ** np.arange(lmin - 1.0, lmax + 2.0)
    else:
        majors = list(np.unique(majors))
        while np.min(majors) > min:
            majors.insert(0, majors[0] / (majors[1] / majors[0]))
        while np.max(majors) < max:
            majors.append(majors[-1] / (majors[-2] / majors[-1]))

    ranges = []
    x1 = majors[0]
    for x2 in majors[1:]:
        ranges.append(
            np.array(
                list(filter(lambda x: min < x < max, np.linspace(x1, x2, n)[1:-1]))
            )
        )
        x1 = x2

    return np.concatenate(ranges)


def plot(select_pid, pid_label: str, xidx: int):
    data = collect_data()
    x = o["interpolation_xgrid"][xidx]
    q2s = list(data[0].keys())
    # ks = np.array([get_theory(tid)["kbThr"] for tid in [0, 1,2]])
    # scale = get_theory(0)["mb"]
    # major_q2s = (ks * scale) ** 2
    # minor_q2s = log_minor_ticks(np.min(q2s), np.max(q2s), majors=major_q2s)

    fig = plt.figure()
    fig.suptitle(f"LHA, toy, pid={pid_label}, x = {x}")
    ax = fig.add_subplot(111)
    ax.hlines(0.0, np.min(q2s), np.max(q2s), colors="#bbbbbb", linestyles="dashed")

    baseline = np.array([select_pid(el)[xidx] for el in data[0].values()])
    for tid in [1, 2]:
        kbThr = get_theory(tid)["kbThr"]
        dat = np.array([select_pid(el)[xidx] for el in data[tid].values()])
        ax.plot(q2s, dat / baseline - 1.0, label=kbThr)

    ax.set_ylabel("rel. distance to µ=1")
    ax.set_xscale("log")
    # ax.set_xlim([np.min(q2s), np.max(q2s)])
    # ax.xaxis.set_ticks(major_q2s, labels=("$m_b^2/2$", "$m_b^2$", "$2 m_b^2$"))
    # ax.xaxis.set_ticks(minor_q2s, minor=True)
    # # ax.xaxis.set_ticklabels(("$m_b^2/2$", "$m_b^2$", "$2 m_b^2$"))
    # ax.grid(which="minor", axis="both", color=".9", linewidth=0.6, linestyle="--")
    # ax.tick_params(labelbottom=True, which="major")
    ax.set_xlabel(r"$\mu^2$ [GeV²]")

    ax.legend()
    fig.savefig(f"check-matching-pid_{pid_label}-xidx_{xidx}.pdf")
    plt.close(fig)


# compute()
plot(lambda s: s[21], 21, 10)
