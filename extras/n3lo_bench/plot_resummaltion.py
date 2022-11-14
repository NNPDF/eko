import pathlib
import matplotlib.pyplot as plt
import numpy as np
from eko.constants import CA, CF
from matplotlib import cm

from splitting_function_utils import splitting_function, compute_a_s
import utils

plt.style.use(utils.load_style())

nf = 4

def load_xgrid():
    with open("small-x/P_as020_nf4.dat", encoding="utf-8") as file:
        raw_data = np.loadtxt(file)
    xgrid = raw_data[:, 1]
    return xgrid

def check_resummed_splitting(g_lo, g_nlo, g_nnlo, entry):
    with open("small-x/P_as020_nf4.dat", encoding="utf-8") as file:
        raw_data = np.loadtxt(file)

    if entry == "gg":
        xPLO = raw_data[:, 5]
        xPNLO = raw_data[:, 9]
        xPNNLO = raw_data[:, 13]
    elif entry == "gq":
        xPLO = raw_data[:, 6]
        xPNLO = raw_data[:, 10]
        xPNNLO = raw_data[:, 14]
    elif entry == "qg":
        xPLO = raw_data[:, 7]
        xPNLO = raw_data[:, 11]
        xPNNLO = raw_data[:, 15]
    elif entry == "qq":
        xPLO = raw_data[:, 8]
        xPNLO = raw_data[:, 12]
        xPNNLO = raw_data[:, 16]

    # ponts are large x are different
    np.testing.assert_allclose(g_lo[:-2], xPLO[:-2], atol=1e-4)
    np.testing.assert_allclose(g_nlo[:-2], xPNLO[:-2], atol=4e-3)
    np.testing.assert_allclose(g_nnlo[:-2], xPNNLO[:-2], atol=4e-3)


def resummed_splitting_xpx(entry):
    with open("small-x/P_as020_nf4.dat", encoding="utf-8") as file:
        raw_data = np.loadtxt(file)

    # LO
    xd0P_gg = raw_data[:, 2]
    xd0P_qg = np.zeros_like(xd0P_gg)
    xd1P_gg = raw_data[:, 3]
    xd1P_qg = raw_data[:, 4]
    # NNLO
    xdP12_gg = raw_data[:, 31]
    xdP12_qg = raw_data[:, 32]
    xd2P_gg = xd1P_gg - xdP12_gg
    xd2P_qg = xd1P_qg - xdP12_qg
    # N3LO
    xd3P_gg = raw_data[:, 42]
    xd3P_qg = raw_data[:, 43]

    if entry == "gg":
        return np.array([xd0P_gg, xd1P_gg, xd2P_gg, xd3P_gg]).T
    if entry == "qg":
        return np.array([xd0P_qg, xd1P_qg, xd2P_qg, xd3P_qg]).T
    if entry == "gq":
        return np.array([xd0P_gg, xd1P_gg, xd2P_gg, xd3P_gg]).T * CF / CA
    if entry == "qq":
        return np.array([xd0P_qg, xd1P_qg, xd2P_qg, xd3P_qg]).T * CF / CA
    raise NotImplementedError(f"{entry} not found")


def p3_resummed(entry):
    with open("small-x/P_as020_nf4.dat", encoding="utf-8") as file:
        raw_data = np.loadtxt(file)

    xP3_gg = raw_data[:, 38]
    xP3_qg = raw_data[:, 39]
    if entry == "gg":
        return xP3_gg
    if entry == "qg":
        return xP3_qg
    if entry == "gq":
        return xP3_gg * CF / CA
    if entry == "qq":
        return xP3_qg * CF / CA
    raise NotImplementedError(f"{entry} not found")


def plot_ad(entry):

    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(5, 1)
    ax = plt.subplot(gs[:, 0])

    g = splitting_function(entry, x_grid, nf)
    a_s = compute_a_s()
    g_lo = g[:, 0] * a_s
    g_nlo = g_lo + g[:, 1] * a_s**2
    g_nnlo = g_nlo + g[:, 2] * a_s**3
    g_n3lo = g_nnlo + g[:, 3] * a_s**4

    check_resummed_splitting(g_lo, g_nlo, g_nnlo, entry)

    g_resumm = resummed_splitting_xpx(entry)
    g_ll = g_resumm[:, 0]
    g1_nll = g_resumm[:, 1]
    g2_nll = g_resumm[:, 2]
    g3_nll = g_resumm[:, 3]

    ax.plot(x_grid, g_n3lo + g3_nll, label="N3LO + NLL", color=cm.get_cmap("tab20")(0))
    ax.plot(x_grid, g_n3lo, label="N3LO", color=cm.get_cmap("tab20")(1))

    ax.plot(
        x_grid,
        g_nnlo + g2_nll,
        linestyle="dashed",
        label="NNLO + NLL",
        color=cm.get_cmap("tab20")(2),
    )
    ax.plot(
        x_grid, g_nnlo, linestyle="dashed", label="NNLO", color=cm.get_cmap("tab20")(3)
    )
    ax.plot(
        x_grid,
        g_nlo + g1_nll,
        linestyle="dashdot",
        label="NLO + NLL",
        color=cm.get_cmap("tab20")(4),
    )
    # ax.plot(
    #     x_grid, g_nlo, linestyle="dashdot", label="NLO", color=cm.get_cmap("tab20")(5)
    # )
    ax.plot(
        x_grid,
        g_lo + g_ll,
        linestyle="dotted",
        label="LO + LL",
        color=cm.get_cmap("tab20")(6),
    )
    # ax.plot(x_grid, g_lo, linestyle="dotted", label="LO", color=cm.get_cmap("tab20")(7))

    x_grid_min, x_grid_max = x_grid.min(), x_grid.max()
    ax.set_xscale("log")
    ax.set_ylim(top=g_n3lo[0] + g3_nll[0])
    ax.set_xlim(x_grid_min, x_grid_max)
    variable = "x"
    ax.set_xlabel(f"${variable}$")
    title = (
        "$x P_{"
        + entry
        + r"}("
        + variable
        + r"), \ \alpha_s=$"
        + str(np.round(a_s * 4 * np.pi, 3))
        + r"$\ n_f=$"
        + str(nf)
    )
    line_value = 0
    ax.hlines(
        line_value,
        x_grid_min,
        x_grid_max,
        linestyles="dotted",
        color="black",
        linewidth=0.5,
    )
    plt.title(title)
    ax.legend()

    # save
    plt.tight_layout()
    plt.savefig(f"small-x/gamma_{entry}_resummed.pdf")


def plot_ad_check(entry):

    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(5, 1)
    ax = plt.subplot(gs[:, 0])

    g = splitting_function(entry, x_grid, nf)
    a_s = compute_a_s()
    g_lo = g[:, 0] * a_s
    g_nlo = g_lo + g[:, 1] * a_s**2
    g_nnlo = g_nlo + g[:, 2] * a_s**3
    g_n3lo = g_nnlo + g[:, 3] * a_s**4

    g_resumm = resummed_splitting_xpx(entry)
    g2_nll = g_resumm[:, 2]
    g3_nll = g_resumm[:, 3]

    ax.plot(x_grid, g_n3lo + g3_nll, label="N3LO + NLL", color=cm.get_cmap("tab20")(0))
    ax.plot(x_grid, g_n3lo, label="N3LO", color=cm.get_cmap("tab20")(1))
    ax.plot(
        x_grid,
        g_nnlo + p3_resummed(entry),
        label="N3LO app (NNLO + dP2 - dP3)",
        color=cm.get_cmap("tab20")(4),
    )
    ax.plot(
        x_grid,
        g_nnlo + g2_nll,
        linestyle="dashed",
        label="NNLO + NLL",
        color=cm.get_cmap("tab20")(2),
    )
    ax.plot(
        x_grid, g_nnlo, linestyle="dashed", label="NNLO", color=cm.get_cmap("tab20")(3)
    )

    x_grid_min, x_grid_max = x_grid.min(), x_grid.max()
    ax.set_xscale("log")
    ax.set_ylim(top=g_n3lo[0] + g3_nll[0])
    ax.set_xlim(x_grid_min, x_grid_max)
    variable = "x"
    ax.set_xlabel(f"${variable}$")
    title = (
        "$x P_{"
        + entry
        + r"}("
        + variable
        + r"), \ \alpha_s=$"
        + str(np.round(a_s * 4 * np.pi, 3))
        + r"$\ n_f=$"
        + str(nf)
    )
    line_value = 0
    ax.hlines(
        line_value,
        x_grid_min,
        x_grid_max,
        linestyles="dotted",
        color="black",
        linewidth=0.5,
    )
    plt.title(title)
    ax.legend()

    # save
    plt.tight_layout()
    plt.savefig(f"small-x/gamma_{entry}_bench.pdf")


if __name__ == "__main__":

    for k in ["gq"]:  # ["gg", "qg", "qq", "gq", "qg"]:
        x_grid = load_xgrid()
        # plot_ad(k)
        plot_ad_check(k)
