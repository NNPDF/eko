import numpy as np
from click import progressbar
from scipy import integrate

from eko import scale_variations as sv
from eko.couplings import CouplingEvolutionMethod, Couplings, CouplingsInfo
from eko.mellin import Path
from eko.quantities.heavy_quarks import QuarkMassScheme
from ekore.anomalous_dimensions.unpolarized.space_like import gamma_ns, gamma_singlet

map_singlet_entries = {"gg": (1, 1), "gq": (1, 0), "qg": (0, 1), "qq": (0, 0)}
map_non_singlet_modes = {"+": 10101, "-": 10201, "v": 10200}


def compute_ad(nf, n_grid, ns_mode=None, n3lo_variation=(0, 0, 0, 0, 0, 0, 0)):
    ns_mode = map_non_singlet_modes.get(ns_mode, None)
    gs_list = []
    for n in n_grid:
        if ns_mode is not None:
            gs_list.append(gamma_ns((4, 0), ns_mode, n, nf, n3lo_variation).real)
        else:
            gs_list.append(gamma_singlet((4, 0), n, nf, n3lo_variation).real)
    return np.array(gs_list)


def non_singlet_ad(entry, n_grid, nf, full_ad=False):
    ns_mode = entry[-1]
    gamma = compute_ad(nf=nf, n_grid=n_grid, ns_mode=ns_mode)
    if full_ad:
        return gamma
    return gamma[:, 1], gamma[:, 2], gamma[:, 3]


def singlet_ad(entry, n_grid, nf, full_ad=False):
    idx1, idx2 = map_singlet_entries[entry]
    gamma = compute_ad(nf, n_grid)
    if full_ad:
        return gamma[:, :, idx1, idx2]
    return gamma[:, 1, idx1, idx2], gamma[:, 2, idx1, idx2], gamma[:, 3, idx1, idx2]


def integrand(u, x, order, entry, nf, ns_mode, n3lo_variation, L):
    is_singlet = ns_mode is None
    path = Path(u, np.log(x), is_singlet)
    integrand = path.prefactor * x ** (-path.n) * path.jac
    if integrand == 0.0:
        return 0.0

    if is_singlet:
        gamma = gamma_singlet((order + 1, 0), path.n, nf, n3lo_variation)
        if L != 0:
            gamma = sv.exponentiated.gamma_variation(gamma, (order + 1, 0), nf, L)
        idx1, idx2 = map_singlet_entries[entry]
        gamma = gamma[order, idx1, idx2]
    else:
        gamma = gamma_ns((order + 1, 0), ns_mode, path.n, nf, n3lo_variation)
        if L != 0:
            gamma = sv.exponentiated.gamma_variation(gamma, (order + 1, 0), nf, L)
        gamma = gamma[order]

    # recombine everything
    return np.real(gamma * integrand)


def splitting_function(
    entry, x_grid, nf, n3lo_variation=(0, 0, 0, 0), orders=None, L=0
):
    """Compute the x-space splitting function, returns x P(x)"""
    mellin_cut = 5e-2
    ns_mode = map_non_singlet_modes.get(entry, None)
    gamma_x = []
    # loop on pto
    orders = [0, 1, 2, 3] if orders is None else orders
    for order in orders:
        print(
            f"Computing splitting function {entry} @ pto: {order}, variation: {n3lo_variation}"
        )
        tot_res = []
        # loop on xgrid
        with progressbar(x_grid) as bar:
            for x in bar:
                if x == 1:
                    tot_res.append(0)
                    continue
                res = integrate.quad(
                    lambda u: integrand(
                        u, x, order, entry, nf, ns_mode, n3lo_variation, L
                    ),
                    0.5,
                    1.0 - mellin_cut,
                    epsabs=1e-12,
                    epsrel=1e-6,
                    limit=200,
                    full_output=1,
                )[0]
                tot_res.append(-x * res)
            gamma_x.append(tot_res)
    return np.array(gamma_x).T


def compute_a_s(q2=None, xif2=1.0, nf=None, order=(4, 0)):
    if q2 is not None:
        ref = CouplingsInfo(
            alphas=0.1181,
            alphaem=0.007496,
            ref=(91.00, 5),
        )
        sc = Couplings(
            couplings=ref,
            order=order,
            method=CouplingEvolutionMethod.EXPANDED,
            masses=np.array([1.51, 4.92, 172.5]) ** 2,
            hqm_scheme=QuarkMassScheme.POLE,
            thresholds_ratios=np.array([1.0, 1.0, 1.0]) ** 2 * xif2,
        )
        return sc.a_s(scale_to=q2 * xif2, nf_to=nf)
    return 0.2 / (4 * np.pi)
