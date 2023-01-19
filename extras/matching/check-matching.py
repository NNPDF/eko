import copy
import logging
import pathlib
import sys

import matplotlib.colors as clr
import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import utils
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
            x = evolution_operator.metadata.rotations.targetgrid.raw
            data[id] = {
                q2: el["pdfs"] for q2, el in apply_pdf(evolution_operator, pdf).items()
            }
    return data


plt.style.use(utils.load_style("./style.yaml"))
hatches = ["////", "\\\\", "..", "+++", "OO"]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot(data, plots, fn, title):
    ptos = data.keys()
    Q2s = plots["Q2s"]
    major_q2s = (plots["ks"] * plots["scale"]) ** 2
    minor_q2s = utils.log_minor_ticks(np.min(Q2s), np.max(Q2s), majors=major_q2s)

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(title)

    gs = matplotlib.gridspec.GridSpec(3, 2, width_ratios=[2, 3], figure=fig)

    ax0 = fig.add_subplot(gs[:, 0])
    ax0plots = {}
    for pto, hatch, color in zip(ptos, hatches, colors):
        p = {}
        p["line"] = ax0.semilogx(Q2s, data[pto][1], color=color)
        # ax.plot(Q2s, data[pto][0],"x")
        # ax.plot(Q2s, data[pto][2],"v")
        p["fill"] = ax0.fill_between(
            Q2s,
            data[pto][0],
            data[pto][2],
            facecolor=clr.to_rgba(color, alpha=0.1),
            label=pto,
            hatch=hatch,
            edgecolor=clr.to_rgba(color, alpha=0.4),
        )
        ax0plots[pto] = p

    ax0.set_xlim([np.min(Q2s), np.max(Q2s)])
    ax0.xaxis.set_ticks(major_q2s)
    ax0.xaxis.set_ticks(minor_q2s, minor=True)
    ax0.xaxis.set_ticklabels(("$m_b^2/4$", "$m_b^2$", "$4 m_b^2$"))
    ax0.tick_params(which="minor", bottom=True, labelbottom=False)
    ax0.grid(which="minor", axis="both", color=".9", linewidth=0.6, linestyle="--")
    ax0.legend([(p["line"][0], p["fill"]) for p in ax0plots.values()], ax0plots.keys())
    ax0.set_ylabel(plots["ylabel"])
    fig.supxlabel(plots["xlabel"])

    ptoaxes = []
    for i, (pto, hatch, color) in enumerate(zip(ptos, hatches, colors)):
        ax = fig.add_subplot(gs[i, 1])
        ptoaxes.append(ax)

        ax.plot(Q2s, data[pto][1] / data[pto][1] - 1.0, color=color)
        ax.fill_between(
            Q2s,
            data[pto][0] / data[pto][1] - 1.0,
            data[pto][2] / data[pto][1] - 1.0,
            facecolor=clr.to_rgba(color, alpha=0.1),
            label=pto,
            hatch=hatch,
            edgecolor=clr.to_rgba(color, alpha=0.4),
        )
        ax.set_xscale("log")
        ax.set_xlim([np.min(Q2s), np.max(Q2s)])
        if "ylim" in plots["breakout"]:
            ax.set_ylim(plots["breakout"]["ylim"])
        ax.yaxis.tick_right()
        ax.tick_params(
            which="both",
            bottom=True,
            top=True,
            left=True,
            right=True,
            labelbottom=False,
        )
        ax.xaxis.set_ticks(major_q2s)
        ax.xaxis.set_ticks(minor_q2s, minor=True)
        ax.xaxis.set_ticklabels(("$m_b^2/4$", "$m_b^2$", "$4 m_b^2$"))
        ax.grid(which="minor", axis="both", color=".9", linewidth=0.6, linestyle="--")

    ptoaxes[-1].tick_params(labelbottom=True, which="major")

    fig.tight_layout(pad=0.8)
    fig.savefig(fn)


def dump(select_pid, pid_label: str, xidx: int):
    data = collect_data()
    x = o["interpolation_xgrid"][xidx]
    q2s = list(data[0].keys())
    scale = get_theory(0)["mb"]
    ks = np.array([get_theory(tid)["kbThr"] for tid in [1, 0, 2]])
    baseline = np.array([select_pid(el)[xidx] for el in data[0].values()])
    plot_data = {}
    for pto, tids in [
        ("LO", [1, 0, 2]),
        # ("NLO", [4, 3, 5]), ("NNLO", [7, 6, 8])
    ]:
        lo = []
        for tid in tids:
            kbThr = get_theory(tid)["kbThr"]
            lo.append(np.array([x * select_pid(el)[xidx] for el in data[tid].values()]))
        plot_data[pto] = lo
    plot_cfg = dict(
        ks=ks,
        Q2s=q2s,
        scale=scale,
        xlabel=r"$\mu^2$",
        ylabel=f"x{pid_label}(x={x:.3e})",
        breakout={},
    )
    plot(
        plot_data,
        plot_cfg,
        f"check-matching-pid_{pid_label}-xidx_{xidx}.pdf",
        f"LHA, toy, x{pid_label}(x={x:.3e},$\\mu^2$)",
    )


# compute()


def select_g(s):
    return s[21]


def select_u(s):
    return s[2]


def select_b(s):
    return s[5]


def select_t3(s):
    return s[2] + s[-2] - (s[1] + s[-1])  # T3 = u+ - d+


for lab, fnc in [("g", select_g), ("u", select_u), ("b", select_b), ("T3", select_t3)]:
    dump(fnc, lab, 10)
    dump(fnc, lab, 20)
    dump(fnc, lab, 40)
