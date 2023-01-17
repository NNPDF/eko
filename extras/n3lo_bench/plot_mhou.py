import pathlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from eko.interpolation import make_lambert_grid
from eko.constants import CA, CF

from splitting_function_utils import compute_a_s, splitting_function
from plot_msht import build_n3lo_var, n3lo_vars_dict
import utils

plt.style.use(utils.load_style())


def build_gamma(g, order, q2, fact_sale, nf):
    a_s = compute_a_s(q2, fact_scale=fact_sale, order=(order + 1, 0), nf=nf)
    gamma = 0
    for pto in range(order + 1):
        gamma += g[:, pto] * a_s ** (pto + 1)
    return gamma


def compute_mhou(g_min, g_max, g_central, q2, xif2_low, xif2_hig, order, nf):

    gamma_low = build_gamma(g_min, order, q2 / xif2_low, q2, nf)
    gamma_hig = build_gamma(g_max, order, q2 / xif2_hig, q2, nf)
    delta_low = gamma_low - g_central
    delta_hig = gamma_hig - g_central
    mhou = 0.5 * np.sqrt(delta_low**2 + delta_hig**2)
    return mhou


def plot_ad(entry, q2=None, nf=4, logscale=True, plot_totu=True, plot_scaling=False):
    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(5, 1)
    ax = plt.subplot(gs[:, 0])

    grid = x_grid[:-2] if entry != "qq" else x_grid[:-3]
    g = splitting_function(entry, grid, nf)
    # compute scale variations
    xif2_low = 0.5**2
    xif2_hig = 2**2
    g_low = splitting_function(entry, grid, nf, L=np.log(xif2_low))
    g_hig = splitting_function(entry, grid, nf, L=np.log(xif2_hig))

    a_s_n3lo = compute_a_s(q2, fact_scale=q2, order=(4, 0), nf=nf)
    g_lo = build_gamma(g, 0, q2, q2, nf)
    g_nlo = build_gamma(g, 1, q2, q2, nf)
    g_nnlo = build_gamma(g, 2, q2, q2, nf)
    g_n3lo = build_gamma(g, 3, q2, q2, nf)

    g_n3lo_var = []
    if plot_totu:
        for idx in range(1, n3lo_vars_dict[entry] + 1):
            var = build_n3lo_var(entry, idx)
            g_n3lo_var.append(splitting_function(entry, grid, nf, var, [3])[:, 0])
        g_n3lo_var = np.array(g_n3lo_var)
        ihou_n3lo = g_n3lo_var.std(axis=0) * a_s_n3lo**4

    mhou_lo = compute_mhou(g_low, g_hig, g_lo, q2, xif2_low, xif2_hig, 0, nf)
    mhou_nlo = compute_mhou(g_low, g_hig, g_nlo, q2, xif2_low, xif2_hig, 1, nf)
    mhou_nnlo = compute_mhou(g_low, g_hig, g_nnlo, q2, xif2_low, xif2_hig, 2, nf)
    mhou_n3lo = compute_mhou(g_low, g_hig, g_n3lo, q2, xif2_low, xif2_hig, 3, nf)

    ax.plot(grid, g_n3lo, label="N3LO ((MHOU) + IHOU)" if plot_totu else "N3LO (MHOU)")
    ax.fill_between(
        grid,
        g_n3lo - mhou_n3lo,
        g_n3lo + mhou_n3lo,
        alpha=0.2,
    )
    if plot_totu:
        n3lo_totu = np.sqrt(ihou_n3lo**2 + mhou_n3lo**2)
        ax.fill_between(
            grid,
            g_n3lo - n3lo_totu,
            g_n3lo + n3lo_totu,
            alpha=0.4,
            color=cm.get_cmap("tab20c")(2),
        )

    ax.plot(grid, g_nnlo, linestyle="dashed", label="NNLO (MHOU)")
    ax.fill_between(
        grid,
        g_nnlo - mhou_nnlo,
        g_nnlo + mhou_nnlo,
        alpha=0.2,
    )
    ax.plot(grid, g_nlo, linestyle="dashdot", label="NLO (MHOU)")
    ax.fill_between(
        grid,
        g_nlo - mhou_nlo,
        g_nlo + mhou_nlo,
        alpha=0.2,
    )
    ax.plot(grid, g_lo, linestyle="dotted", label="LO (MHOU)")
    ax.fill_between(
        grid,
        g_lo - mhou_lo,
        g_lo + mhou_lo,
        alpha=0.2,
    )

    if plot_scaling:
        if entry == "gq":
            gg = splitting_function("gg", grid, nf, (0, 0, 0, 0))
            gg_lo = gg[:, 0] * a_s_n3lo
            gg_nlo = gg_lo + gg[:, 1] * a_s_n3lo**2
            gg_nnlo = gg_nlo + gg[:, 2] * a_s_n3lo**3
            gg_n3lo = gg_nnlo + gg[:, 3] * a_s_n3lo**4
            ax.plot(
                grid[:10],
                gg_n3lo[:10] * CF / CA,
                linestyle="dashdot",
                color="black",
                label=r"$C_f/C_A P_{gg}$",
                alpha=0.5,
            )
        elif entry == "qg":
            qq = splitting_function("qq", grid, nf, (0, 0, 0, 0))
            qq_lo = qq[:, 0] * a_s_n3lo
            qq_nlo = qq_lo + qq[:, 1] * a_s_n3lo**2
            qq_nnlo = qq_nlo + qq[:, 2] * a_s_n3lo**3
            qq_n3lo = qq_nnlo + qq[:, 3] * a_s_n3lo**4
            ax.plot(
                grid[:10],
                qq_n3lo[:10] * CA / CF,
                linestyle="dashdot",
                color="black",
                label=r"$C_A/C_f P_{qq}$",
                alpha=0.5,
            )

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
        + f"{(a_s_n3lo * 4 * np.pi):.3f}"
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
    pathlib.Path(utils.here / "compare_mhou").mkdir(parents=True, exist_ok=True)
    unc = "totu" if plot_totu else "mhou"
    xscale = "logx" if logscale else "linx"
    plt.savefig(f"compare_mhou/gamma_{entry}_{unc}_{xscale}.pdf")


def plot_ad_ratio(entry, q2=None, nf=4, plot_totu=False):
    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(5, 1)
    ax = plt.subplot(gs[:, 0])

    grid = x_grid[:-1] if entry != "qq" else x_grid[:-3]
    g = splitting_function(entry, grid, nf)
    # compute scale variations
    xif2_low = 0.5**2
    xif2_hig = 2**2
    g_low = splitting_function(entry, grid, nf, L=np.log(xif2_low))
    g_hig = splitting_function(entry, grid, nf, L=np.log(xif2_hig))

    a_s_n3lo = compute_a_s(q2, fact_scale=q2, order=(4, 0), nf=nf)
    g_nlo = build_gamma(g, 1, q2, q2, nf)
    g_nnlo = build_gamma(g, 2, q2, q2, nf)
    g_n3lo = build_gamma(g, 3, q2, q2, nf)

    g_n3lo_var = []
    if plot_totu:
        for idx in range(1, n3lo_vars_dict[entry] + 1):
            var = build_n3lo_var(entry, idx)
            g_n3lo_var.append(splitting_function(entry, grid, nf, var, [3])[:, 0])
        g_n3lo_var = np.array(g_n3lo_var)
        ihou_n3lo = g_n3lo_var.std(axis=0) * a_s_n3lo**4

    mhou_nlo = compute_mhou(g_low, g_hig, g_nnlo, q2, xif2_low, xif2_hig, 1, nf)
    mhou_nnlo = compute_mhou(g_low, g_hig, g_nnlo, q2, xif2_low, xif2_hig, 2, nf)
    mhou_n3lo = compute_mhou(g_low, g_hig, g_n3lo, q2, xif2_low, xif2_hig, 3, nf)

    ax.plot(
        grid,
        g_n3lo / g_nlo,
        label="N3LO ((MHOU) + IHOU)" if plot_totu else "N3LO (MHOU)",
    )
    ax.fill_between(
        grid,
        (g_n3lo - mhou_n3lo) / g_nlo,
        (g_n3lo + mhou_n3lo) / g_nlo,
        alpha=0.2,
    )
    if plot_totu:
        n3lo_totu = np.sqrt(ihou_n3lo**2 + mhou_n3lo**2)
        ax.fill_between(
            grid,
            (g_n3lo - n3lo_totu) / g_nlo,
            (g_n3lo + n3lo_totu) / g_nlo,
            alpha=0.4,
            color=cm.get_cmap("tab20c")(2),
        )

    ax.plot(grid, g_nnlo / g_nlo, linestyle="dashed", label="NNLO (MHOU)")
    ax.fill_between(
        grid,
        (g_nnlo - mhou_nnlo) / g_nlo,
        (g_nnlo + mhou_nnlo) / g_nlo,
        alpha=0.2,
    )
    ax.fill_between(
        grid,
        1 - mhou_nlo / g_nlo,
        1 + mhou_nlo / g_nlo,
        alpha=0.2,
    )

    grid_min, grid_max = grid.min(), grid.max()
    ax.set_xlim(grid_min, grid_max)
    if entry in ["gg", "gq"]:
        ax.set_ylim(0.8, 1.1)
    variable = "x"
    ax.set_xlabel(f"${variable}$")
    ax.set_ylabel(r"$\rm Ratio\ to\ NLO$")
    title = (
        "$x P_{"
        + entry
        + r"}("
        + variable
        + r"), \ \alpha_s=$"
        + str(np.round(a_s_n3lo * 4 * np.pi, 3))
        + r"$\ n_f=$"
        + str(nf)
    )
    line_value = 1
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
    pathlib.Path(utils.here / "compare_mhou").mkdir(parents=True, exist_ok=True)
    unc = "totu" if plot_totu else "mhou"
    plt.savefig(f"compare_mhou/gamma_{entry}_{unc}_ratio.pdf")


if __name__ == "__main__":

    for k in ["qg", "gq", "gg", "qq"]:

        # linear plots
        q2 = 38.5  # chosen such that a_s(Q2) \approx 0.2
        nf = 5
        plot_totu = True
        x_grid = make_lambert_grid(80, x_min=1e-2)
        # plot_ad(k, q2=q2, nf=nf, logscale=False, plot_totu=plot_totu)
        plot_ad_ratio(k, q2=q2, nf=nf, plot_totu=plot_totu)

        # log plots
        x_grid = make_lambert_grid(80, x_min=1e-7)
        # plot_ad(k, q2=q2, nf=nf, plot_scaling=True, plot_totu=plot_totu)
