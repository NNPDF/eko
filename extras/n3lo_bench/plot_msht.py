import pathlib
import matplotlib.pyplot as plt
import msht_n3lo as msht
import numpy as np
from eko.interpolation import make_lambert_grid

from splitting_function_utils import compute_a_s, splitting_function
from small_x_limit import singlet_to_0
import utils

plt.style.use(utils.load_style())


n3lo_vars_dict = {
    "qq": 17,
    "gg": 18,
    "gq": 24,
    "qg": 21,
}


def build_n3lo_var(entry, idx):
    if entry == "gg":
        return (idx, 0, 0, 0)
    elif entry == "gq":
        return (0, idx, 0, 0)
    elif entry == "qg":
        return (0, 0, idx, 0)
    elif entry == "qq":
        return (0, 0, 0, idx)
    return (0, 0, 0, 0)


def msht_splitting(entry, x, nf):

    if entry == "gg":
        a1, a2 = -5, 15
        return msht.pgg3a(x, a1), msht.pgg3a(x, a2)
    if entry == "gq":
        a1, a2 = -1.8, -1.5
        return msht.pgq3a(x, a1), msht.pgq3a(x, a2)
    if entry == "qg":
        a1, a2 = -2.5, -0.8
        return nf * msht.pqg3a(x, a1), nf * msht.pqg3a(x, a2)
    if entry == "qq":
        aps1, aps2 = -0.7, 0
        ans1, ans2 = 0, 0.014
        qqps = np.array([msht.pqqps3a(x, aps1), msht.pqqps3a(x, aps2)])
        qqns = np.array([msht.p3nsa(x, ans1, 1, nf), msht.p3nsa(x, ans2, 1, nf)])
        qq = nf * qqps + qqns
        return qq[0], qq[1]
    raise ValueError(f"Entry not found {entry}")


def msht_splitting_xpx(entry, grid, nf):
    msht_grid_min = []
    msht_grid_max = []
    for x in grid:
        g1, g2 = msht_splitting(entry, x, nf)
        msht_grid_min.append(x * g1)
        msht_grid_max.append(x * g2)
    return np.array(msht_grid_min), np.array(msht_grid_max)


def plot_ad(entry, q2=None, nf=4, logscale=True, show_candidates=False):
    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(5, 1)
    ax = plt.subplot(gs[:, 0])

    grid = x_grid[:-2] if entry != "qq" else x_grid[:-3]
    g = splitting_function(entry, grid, nf)
    small_x = singlet_to_0(entry, grid, nf)

    a_s = compute_a_s(q2)
    g_lo = g[:, 0] * a_s
    g_nlo = g_lo + g[:, 1] * a_s**2
    g_nnlo = g_nlo + g[:, 2] * a_s**3
    g_n3lo = g_nnlo + g[:, 3] * a_s**4
    small_x = small_x * a_s**4

    g_n3lo_var = []
    for idx in range(1, n3lo_vars_dict[entry] + 1):
        var = build_n3lo_var(entry, idx)
        g_n3lo_var.append(splitting_function(entry, grid, nf, var, [3])[:, 0])
    g_n3lo_var = np.array(g_n3lo_var)
    g_n3lo_min = g_n3lo - g_n3lo_var.std(axis=0) * a_s**4
    g_n3lo_max = g_n3lo + g_n3lo_var.std(axis=0) * a_s**4

    g_msht_n3lo_min, g_msht_n3lo_max = msht_splitting_xpx(entry, grid, nf)
    g_msht_n3lo = (g_msht_n3lo_min + g_msht_n3lo_max) / 2

    g_msht_n3lo_min = g_msht_n3lo_min * a_s**4 + g_nnlo
    g_msht_n3lo_max = g_msht_n3lo_max * a_s**4 + g_nnlo
    g_msht_n3lo = g_msht_n3lo * a_s**4 + g_nnlo

    ax.plot(grid, g_msht_n3lo, label="MSHT@N3LO")
    ax.fill_between(
        grid,
        g_msht_n3lo_min,
        g_msht_n3lo_max,
        alpha=0.2,
    )

    ax.plot(grid, g_n3lo, label="N3LO")
    if show_candidates:
        for var in g_n3lo_var:
            ax.plot(
                grid, g_nnlo + var * a_s**4, alpha=0.5, color="#D62828"
            )
    else:
        ax.fill_between(
            grid,
            g_n3lo_min,
            g_n3lo_max,
            alpha=0.2,
        )
    ax.plot(grid, g_nnlo, linestyle="dashed", label="NNLO")
    ax.plot(grid, g_nlo, linestyle="dashdot", label="NLO")
    ax.plot(grid, g_lo, linestyle="dotted", label="LO")
    if logscale:
        ax.plot(grid[:10], small_x[:10], linestyle="dashdot", color="black", alpha=0.7)

    grid_min, grid_max = grid.min(), grid.max()
    if logscale:
        ax.set_xscale("log")
    ax.set_xlim(grid_min, grid_max)
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
        grid_min,
        grid_max,
        linestyles="dotted",
        color="black",
        linewidth=0.5,
    )
    plt.title(title)
    plt.grid(visible=True, which="major", color="black", linestyle="dotted", alpha=0.2)
    ax.legend()

    # save
    plt.tight_layout()
    pathlib.Path(utils.here/"compare_msht").mkdir(parents=True, exist_ok=True)
    if logscale and not show_candidates:
        plt.savefig(f"compare_msht/gamma_{entry}_msht_logx.pdf")
    elif show_candidates:
        plt.savefig(f"compare_msht/gamma_{entry}_msht_rep.pdf")
    else:
        plt.savefig(f"compare_msht/gamma_{entry}_msht_linx.pdf")


if __name__ == "__main__":

    for k in ["qg", "gq", "gg", "qq"]:

        # linear plots
        x_grid = make_lambert_grid(60, x_min=1e-2)
        # plot_ad(k, logscale=False)

        # log plots
        x_grid = make_lambert_grid(60, x_min=1e-7)
        plot_ad(k)
