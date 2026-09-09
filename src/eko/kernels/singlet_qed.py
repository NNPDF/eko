"""Collection of QED singlet EKOs."""

import numba as nb
import numpy as np

from .. import hell

from ekore import anomalous_dimensions as ad

from .. import beta
from . import EvoMethods


# NOTE (HELL): caching stays ON -- the small-x bridge uses numba
# ExternalFunction symbols, cache-compatible (see eko/hell.py).
@nb.njit(cache=True)
def eko_iterate(
    gamma_singlet, as_list, a_half, nf, order, ev_op_iterations, dim, n=0j, use_hell=False
):
    """Singlet QEDxQCD iterated (exact) EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target strong coupling value
    a0 : float
        initial strong coupling value
    aem_list : float
        electromagnetic coupling values
    nf : int
        number of active flavors
    order : tuple(int,int)
        QCDxQED perturbative orders
    ev_op_iterations : int
        number of evolution steps

    Returns
    -------
    e_s^{order} : numpy.ndarray
        singlet QEDxQCD iterated (exact) EKO
    """
    e = np.identity(dim, np.complex128)
    betaQCD = np.zeros((order[0] + 1, order[1] + 1))
    for i in range(1, order[0] + 1):
        betaQCD[i, 0] = beta.beta_qcd((i + 1, 0), nf)
    betaQCD[1, 1] = beta.beta_qcd((2, 1), nf)
    for step in range(1, ev_op_iterations + 1):
        ah = as_list[step]
        al = as_list[step - 1]
        as_half = a_half[step - 1, 0]
        aem = a_half[step - 1, 1]
        delta_a = ah - al
        gamma = np.zeros((dim, dim), np.complex128)
        betatot = 0
        for i in range(0, order[0] + 1):
            for j in range(0, order[1] + 1):
                betatot += betaQCD[i, j] * as_half ** (i + 1) * aem**j
                gamma += gamma_singlet[i, j] * as_half**i * aem**j
        if use_hell and dim == 4:
            # NOTE (HELL): add the NLL small-x resummed QCD correction,
            # exactly as in the pure-QCD eko_iterate (see
            # eko.kernels.singlet), embedded into the unified
            # (g, gamma, Sigma, Sigma_Delta) block:
            #  * the photon row/column receive nothing (no QCD small-x
            #    resummation of the photon at this order);
            #  * quark production is flavour-democratic (dSigma_u/d also
            #    receives nu/nf resp. nd/nf of the quark-row entries), and
            #    eko's Sigma_Delta = nd/nu Sigma_u - Sigma_d (FlavorSpace
            #    docs: e.g. nf=5, 3/2(u+c) - (d+s+b)) is built precisely
            #    so that a democratic source cancels: nd/nu*(nu/nf) -
            #    nd/nf = 0 -- the Sigma_Delta row receives NOTHING;
            #  * the incoming-quark column is flavour blind (Delta P_xq =
            #    CF/CA Delta P_xg), so only the Sigma column is fed, never
            #    Sigma_Delta.
            # Conventions: HELL's N has its pole at 0 (pass n-1) and dp
            # returns physical splitting-function corrections (subtract,
            # gamma here is -P), matched to fo = order[0]-1 (1 = NLO,
            # 2 = NNLO, 3 = N3LO).
            # NOTE the normalization difference with
            # the pure-QCD eko_iterate: THERE the gamma tower carries one
            # power of a_s less (LO at a_s^0, so the correction is divided
            # by a_half); HERE the tower is the physical gamma(as, aem)
            # (LO QCD at as^1, betatot likewise unshifted), so DeltaP is
            # subtracted directly, with no division.
            dp_vec = hell.dp(
                order[0] - 1, 4.0 * np.pi * as_half, n.real - 1.0, n.imag
            )
            delta_p = np.zeros((dim, dim), dtype=np.complex128)
            # unified ordering: 0 = g, 1 = gamma, 2 = Sigma, 3 = Sigma_Delta
            delta_p[0, 0] = dp_vec[0]  # dPgg
            delta_p[0, 2] = dp_vec[1]  # dPgq
            delta_p[2, 0] = dp_vec[2]  # dPqg
            delta_p[2, 2] = dp_vec[3]  # dPqq
            gamma -= delta_p
        ln = gamma / betatot * delta_a
        ek = np.ascontiguousarray(ad.exp_matrix(ln)[0])
        e = ek @ e
    return e


@nb.njit(cache=True)
def dispatcher(
    order,
    method,
    gamma_singlet,
    as_list,
    a_half,
    nf,
    ev_op_iterations,
    _ev_op_max_order,
    n=0j,
    use_hell=False,
):
    """Determine used kernel and call it.

    Parameters
    ----------
    order : tuple(int,int)
        perturbative order
    method : int
        method
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem_list : numpy.ndarray
        electromagnetic coupling values
    nf : int
        number of active flavors
    ev_op_iterations : int
        number of evolution steps
    ev_op_max_order : tuple(int,int)
        perturbative expansion order of U

    Returns
    -------
    e_s : numpy.ndarray
        singlet EKO
    """
    if method == EvoMethods.ITERATE_EXACT:
        return eko_iterate(
            gamma_singlet, as_list, a_half, nf, order, ev_op_iterations, 4, n, use_hell
        )
    raise NotImplementedError('Only "iterate-exact" is implemented with QED')
