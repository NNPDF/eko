import pathlib

import matplotlib.pyplot as plt
import numpy as np
import utils
from plot_msht import build_n3lo_var, msht_splitting_xpx, n3lo_vars_dict
from splitting_function_utils import compute_a_s, splitting_function

from eko.interpolation import lambertgrid

plt.style.use(utils.load_style())


def plot_ad_ratio(entry, q2=None, nf=4, posterior=False):
    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(5, 1)
    ax = plt.subplot(gs[:, 0])

    grid = x_grid[:-1] if entry != "qq" else x_grid[:-3]
    g = splitting_function(entry, grid, nf)

    a_s = compute_a_s(q2, nf=nf)
    g_lo = g[:, 0] * a_s
    g_nlo = g_lo + g[:, 1] * a_s**2
    g_nnlo = g_nlo + g[:, 2] * a_s**3
    g_n3lo = g_nnlo + g[:, 3] * a_s**4

    g_n3lo_var = []
    for idx in range(1, n3lo_vars_dict[entry] + 1):
        var = build_n3lo_var(entry, idx)
        g_n3lo_var.append(splitting_function(entry, grid, nf, var, [3])[:, 0])

    g_n3lo_var = np.array(g_n3lo_var)
    g_n3lo_min = g_n3lo - g_n3lo_var.std(axis=0) * a_s**4
    g_n3lo_max = g_n3lo + g_n3lo_var.std(axis=0) * a_s**4

    g_msht_n3lo, g_msht_n3lo_min, g_msht_n3lo_max = msht_splitting_xpx(
        entry, grid, nf, posterior=False
    )

    g_msht_n3lo_min = g_msht_n3lo_min * a_s**4 + g_nnlo
    g_msht_n3lo_max = g_msht_n3lo_max * a_s**4 + g_nnlo
    g_msht_n3lo = g_msht_n3lo * a_s**4 + g_nnlo

    ax.plot(grid, g_msht_n3lo / g_nnlo, label="MSHTaN3LO")
    ax.fill_between(
        grid,
        g_msht_n3lo_min / g_nnlo,
        g_msht_n3lo_max / g_nnlo,
        alpha=0.2,
    )

    ax.plot(grid, g_n3lo / g_nnlo, label="aN3LO")
    ax.fill_between(
        grid,
        g_n3lo_min / g_nnlo,
        g_n3lo_max / g_nnlo,
        alpha=0.2,
    )
    if posterior:
        g_msht_n3lo, g_msht_n3lo_min, g_msht_n3lo_max = msht_splitting_xpx(
            entry, grid, nf, posterior
        )
        g_msht_n3lo_min = g_msht_n3lo_min * a_s**4 + g_nnlo
        g_msht_n3lo_max = g_msht_n3lo_max * a_s**4 + g_nnlo
        g_msht_n3lo = g_msht_n3lo * a_s**4 + g_nnlo
        ax.plot(grid, g_msht_n3lo / g_nnlo, label="MSHTaN3LO (posterior)")
        ax.fill_between(
            grid,
            g_msht_n3lo_min / g_nnlo,
            g_msht_n3lo_max / g_nnlo,
            alpha=0.2,
        )
    ax.plot(grid, g_nnlo / g_nlo, linestyle="dashed", label="NNLO")
    # ax.plot(grid, g_nlo / g_lo, linestyle="dashed", label="NLO")

    grid_min, grid_max = grid.min(), grid.max()
    ax.set_xlim(grid_min, grid_max)
    variable = "x"
    ax.set_xlabel(f"${variable}$")
    ax.set_ylabel(r"$\rm Ratio\ to\ previous\ order$")
    # ax.set_ylabel(r"$\rm Ratio\ to\ LO$")
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
    folder = "compare_msht"
    if posterior:
        folder += "_posterior"
    pathlib.Path(utils.here / folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{folder}/gamma_{entry}_msht_ratio_new.pdf")


if __name__ == "__main__":
    for k in ["gg"]:  # ["gq", "gg", "qg", "qq"]:
        # linear plots
        x_grid = lambertgrid(60, x_min=1e-1)
        plot_ad_ratio(k, posterior=True)
