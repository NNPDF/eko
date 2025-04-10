"""Dump a fast x-space grid of the N3LO transition matrix elements.
The output file have the structure: x_grid, nf=3, nf=4, nf=5.
"""

import numpy as np
from click import progressbar
from eko.mellin import Path
from ekore.harmonics import cache as c
from ekore.operator_matrix_elements.unpolarized.space_like import as3
from eko.interpolation import lambertgrid
from scipy import integrate

from large_n_limit import Agg_asymptotic, Aqq_asymptotic

XGRID = lambertgrid(500, 1e-6)
"""X-grid."""

LOG = 0
"""Matching threshold displaced ?"""

MAP_ENTRIES = {
    "gg": (0, 0),
    "gq": (0, 1),
    "qg": (1, 0),
    "qq": (1, 1),
    "Hg": (2, 0),
    "Hq": (2, 1),
    "gH": (0, 2),
    "HH": (2, 2),
    "qq_ns": (0, 0),
}


def compute_ome(nf, n, is_singlet):
    """Get the correct ome from eko."""
    cache = c.reset()
    if is_singlet:
        return as3.A_singlet(n, cache, nf, L=LOG)
    else:
        return as3.A_ns(n, cache, nf, L=LOG)


def compute_xspace_ome(entry, nf, x_grid=XGRID):
    """Compute the x-space transition matrix element, returns A^3(x)."""
    mellin_cut = 5e-2
    is_singlet = "ns" not in entry

    def integrand(u, x):
        """Mellin inversion integrand."""
        path = Path(u, np.log(x), is_singlet)
        integrand = path.prefactor * x ** (-path.n) * path.jac

        # compute the N space ome
        ome_n = compute_ome(nf, path.n, is_singlet)
        idx1, idx2 = MAP_ENTRIES[entry]
        ome_n = ome_n[idx1, idx2]
        # subtract the large-N limit for diagonal terms (ie local and singular bits)
        if entry in ["qq_ns", "qq"]:
            ome_n -= Aqq_asymptotic(path.n, nf)
        elif entry == "gg":
            ome_n -= Agg_asymptotic(path.n, nf)

        # recombine everything
        return np.real(ome_n * integrand)

    ome_x = []
    print(f"Computing operator matrix element {entry} @ pto: 3, nf: {nf}")
    # loop on xgrid
    with progressbar(x_grid) as bar:
        for x in bar:
            res = integrate.quad(
                lambda u: integrand(u, x),
                0.5,
                1.0 - mellin_cut,
                epsabs=1e-12,
                epsrel=1e-6,
                limit=200,
            )[0]
            ome_x.append(res)

    return np.array(ome_x)


def save_files(entry, ome_x, xgrid=XGRID):
    """Write the space reuslt in a txt file."""
    fname = f"x_space/A_{entry}.txt"
    np.savetxt(fname, np.concatenate(([xgrid], np.array(ome_x))).T)


if __name__ == "__main__":
    # non diagonal temrms
    for k in ["qq_ns", "gg", "gq", "qg", "qq", "Hg", "Hq"]:
        result = [compute_xspace_ome(k, nf) for nf in [3, 4, 5]]
        save_files(k, result)
