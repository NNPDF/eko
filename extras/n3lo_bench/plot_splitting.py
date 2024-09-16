import pathlib

import matplotlib.pyplot as plt
import numpy as np
import utils
from splitting_function_utils import (
    compute_a_s,
    non_singlet_ad,
    singlet_ad,
    splitting_function,
)

plt.style.use(utils.load_style())


n_grid = np.linspace(2, 20, 19)
x_grid = np.geomspace(1e-4, 1.0, 50)


def plot_ad_expansion(entry, nf=4, plot_xspace=False, logscale=False):
    """Plot the splitting function coefficient for each order."""
    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(5, 1)
    ax = plt.subplot(gs[:-2, 0])

    if plot_xspace:
        grid = x_grid
        g = splitting_function(entry, grid, nf)
        g_nlo, g_nnlo, g_n3lo = g[:, 1], g[:, 2], g[:, 3]
    else:
        grid = n_grid
        if "ns" in entry:
            g_nlo, g_nnlo, g_n3lo = non_singlet_ad(entry, grid, nf, entry)
        else:
            g_nlo, g_nnlo, g_n3lo = singlet_ad(entry, grid, nf)

    # plot
    ax.plot(grid, g_n3lo / (4 * np.pi) ** 3, "+", label="N3LO")
    ax.plot(grid, g_nnlo / (4 * np.pi) ** 2, "x", label="NNLO")
    ax.plot(grid, g_nlo / (4 * np.pi), "*", label="NLO")
    grid_min, grid_max = grid.min(), grid.max()
    if logscale:
        ax.set_xscale("log")
    if not plot_xspace:
        grid_min, grid_max = grid.min() - 1, grid.max() + 1
        ax.set_xticks(np.linspace(grid.min(), grid.max(), 10))
    ax.set_xlim(grid_min, grid_max)
    variable = "x" if plot_xspace else "N"
    title = "$x P^{(n)}_{" if plot_xspace else r"$\gamma^{(n)}_{"
    title += entry + r"}(" + variable + r"), \ n_f=$" + str(nf)
    ax.set_ylabel(title)
    ax.hlines(
        0,
        grid_min,
        grid_max,
        linestyles="dotted",
        color="black",
        linewidth=0.5,
    )
    ax.legend()

    # ratio plot
    ax_ratio = plt.subplot(gs[-2:, 0])
    ax_ratio.plot(grid, g_n3lo / (4 * np.pi) ** 2 / g_nlo, "+", label="N3LO")
    ax_ratio.plot(grid, g_nnlo / (4 * np.pi) / g_nlo, "x", label="NNLO")
    ax_ratio.set_xlim(grid_min, grid_max)
    ax_ratio.set_ylabel(r"Ratio to NLO")
    ax_ratio.set_xlabel(f"${variable}$")
    ax_ratio.hlines(
        1,
        grid_min,
        grid_max,
        linestyles="dotted",
        color="black",
        linewidth=0.5,
    )
    if logscale:
        ax_ratio.set_xscale("log")
    ax_ratio.legend()

    # save
    plt.tight_layout()
    pathlib.Path(utils.here / "gamma_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"gamma_plots/gamma_{entry}_expansion.pdf")


def plot_ad(entry, q2=None, nf=4, plot_xspace=False, plot_ratio=False, logscale=False):
    """Plot the splitting function at fixed order."""
    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(5, 1)
    ax = plt.subplot(gs[:, 0])

    if plot_xspace:
        grid = x_grid
        g = splitting_function(entry, grid, nf)
    else:
        grid = n_grid
        if "ns" in entry:
            g = non_singlet_ad(entry, grid, nf=nf, full_ad=True)
        else:
            g = singlet_ad(entry, grid, nf=nf, full_ad=True)

    a_s = compute_a_s(q2, nf=nf)
    g_lo = g[:, 0] * a_s
    g_nlo = g_lo + g[:, 1] * a_s**2
    g_nnlo = g_nlo + g[:, 2] * a_s**3
    g_n3lo = g_nnlo + g[:, 3] * a_s**4

    if plot_ratio:
        ax.plot(grid, g_n3lo / g_nlo, "+", label="N3LO")
        ax.plot(grid, g_nnlo / g_nlo, "x", label="NNLO")
    else:
        ax.plot(grid, g_n3lo, "+", label="N3LO")
        ax.plot(grid, g_nnlo, "x", label="NNLO")
        ax.plot(grid, g_nlo, "*", label="NLO")
        ax.plot(grid, g_lo, ".", label="LO")

    grid_min, grid_max = grid.min(), grid.max()
    if logscale:
        ax.set_xscale("log")
    if not plot_xspace:
        grid_min, grid_max = grid.min() - 1, grid.max() + 1
        ax.set_xticks(np.linspace(grid.min(), grid.max(), 10))
    ax.set_xlim(grid_min, grid_max)
    variable = "x" if plot_xspace else "N"
    ax.set_xlabel(f"${variable}$")
    title = "$x P_{" if plot_xspace else r"$\gamma_{"
    title += (
        entry
        + r"}("
        + variable
        + r"), \ \alpha_s=$"
        + str(np.round(a_s * 4 * np.pi, 3))
        + r"$\ n_f=$"
        + str(nf)
    )
    if plot_ratio:
        title = f"Ratio to NLO {title}"
        line_value = 1
    else:
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
    ax.legend()

    # save
    plt.tight_layout()
    pathlib.Path(utils.here / "gamma_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"gamma_plots/gamma_{entry}.pdf")


if __name__ == "__main__":
    for k in ["gg", "qq", "gq", "qg"]:
        # plot_ad_expansion(k, plot_xspace=True, logscale=True)
        plot_ad(k, plot_xspace=True, plot_ratio=False, logscale=True, q2=10)
