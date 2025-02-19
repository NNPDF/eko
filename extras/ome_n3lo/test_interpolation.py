"""Here we test that integrating the x-space expressions we are indeed reproducing eko."""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from convert_ome_xspace import LOG, MAP_ENTRIES, compute_ome
from large_n_limit import Agg_asymptotic, Aqq_asymptotic


def ome_regular(entry, nf):
    grid = np.loadtxt(f"x_space/A_{entry}.txt")
    return CubicSpline(grid[:, 0], grid[:, nf - 2])


def ome_local(entry, nf):
    if entry == "gg":
        return Agg_asymptotic(0, nf)
    elif entry in ["qq_ns", "qq"]:
        return Aqq_asymptotic(0, nf)
    return 0


def ome_singular(x, entry, nf):
    local = ome_local(entry, nf)
    if entry == "gg":
        asy = Agg_asymptotic(1, nf)
    elif entry in ["qq_ns", "qq"]:
        asy = Aqq_asymptotic(1, nf)

    singular = np.real((asy - local))
    return singular / (- 1 + x)


def test_moments(entry, N, nf):
    # compute N space ome form eko
    is_singlet = "ns" not in entry

    ome_n = compute_ome(nf, complex(N), is_singlet)
    idx1, idx2 = MAP_ENTRIES[entry]
    ome_n = ome_n[idx1, idx2]

    # integrate using the x-space inteprolation
    a_reg = ome_regular(entry, nf)
    ome_x = quad(lambda x: a_reg(x) * x ** (N - 1), 1e-6, 1)[0]

    # add local and singular bits
    if entry in ["qq_ns", "qq", "gg"]:
        # pure plus term
        ome_x += quad(
            lambda x: ome_singular(x, entry, nf) * (x ** (N - 1) - 1), 1e-6, 1
        )[0]
        # # constant part in N space, that should not be needed ?
        # ome_x += quad(
        #     lambda x: - ome_singular(0, entry, nf) * x ** (N - 1), 1e-6, 1
        # )[0]
        ome_x += ome_local(entry, nf)

    np.testing.assert_allclose(ome_n, ome_x, rtol=2e-2, err_msg=f"{entry}, {nf}")


if __name__ == "__main__":
    N = 3.1233
    for nf in [3, 4, 5]:
        # TODO: some entries are passing other no...
        for k in ["qg"]:
        # for k in ["gg", "qq", "qq_ns", "gq", "qg", "Hg", "Hq"]:
            test_moments(k, N, nf)
