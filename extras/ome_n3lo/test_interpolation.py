"""Here we test that integrating the x-space expressions we are indeed reproducing eko."""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from convert_ome_xspace import MAP_ENTRIES, compute_ome
from large_n_limit import Agg_asymptotic, Aqq_asymptotic


def ome_regular(entry, nf):
    grid = np.loadtxt(f"x_space/A_{entry}.txt")
    return CubicSpline(grid[:, 0], grid[:, nf - 2])


def ome_local(entry, nf):
    if entry == "gg":
        return Agg_asymptotic(0, nf).real
    elif entry in ["qq_ns", "qq"]:
        return Aqq_asymptotic(0, nf).real
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
        ome_x += ome_local(entry, nf)

    # TODO: some entries are passing other no...
    # np.testing.assert_allclose(ome_n, ome_x, rtol=4e-2, err_msg=f"{entry}, {nf}")
    return ome_n.real, ome_x


if __name__ == "__main__":
    N = 5.6788
    entries =  ["gg", "qq", "qq_ns", "gq", "qg", "Hg", "Hq"]
    for nf in [3, 4, 5]:
        results = []
        for k in entries:
            results.append(test_moments(k, N, nf))
        df = pd.DataFrame(results, columns=["EKO", "Interpol"], index = entries)
        df["rel_diff"] = ((df.EKO - df.Interpol) / df.EKO)
        print("************************************")
        print(df)
        print("************************************")
