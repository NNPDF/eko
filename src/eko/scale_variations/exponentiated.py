r"""Contains the scale variation for ``ModSV=exponentiated``."""

import numba as nb

from .. import beta


@nb.njit(cache=True)
def gamma_variation(gamma, order, nf, L):
    """Adjust the anomalous dimensions with the scale variations.

    Parameters
    ----------
    gamma : numpy.ndarray
        anomalous dimensions
    order : tuple(int,int)
        perturbation order
    nf : int
        number of active flavors
    L : float
        logarithmic ratio of factorization and renormalization scale

    Returns
    -------
    numpy.ndarray
        adjusted anomalous dimensions
    """
    # since we are modifying *in-place* be careful, that the order matters!
    # and indeed, we need to adjust the highest elements first
    beta0 = beta.beta_qcd((2, 0), nf)
    beta1 = beta.beta_qcd((3, 0), nf)
    beta2 = beta.beta_qcd((4, 0), nf)
    if order[0] >= 4:
        gamma[3] += (
            3.0 * beta0 * L * gamma[2]
            + (2.0 * beta1 * L + 3.0 * beta0**2 * L**2) * gamma[1]
            + (beta2 * L + 5.0 / 2.0 * beta1 * beta0 * L**2 + beta0**3 * L**3)
            * gamma[0]
        )
    if order[0] >= 3:
        gamma[2] += (
            2.0 * beta0 * gamma[1] * L + (beta1 * L + beta0**2 * L**2) * gamma[0]
        )
    if order[0] >= 2:
        gamma[1] += beta0 * gamma[0] * L
    return gamma


@nb.njit(cache=True)
def gamma_variation_qed(gamma, order, nf, nl, L, alphaem_running):
    """Adjust the anomalous dimensions with the scale variations.

    Parameters
    ----------
    gamma : numpy.ndarray
        anomalous dimensions
    order : tuple(int,int)
        perturbation order
    nf : int
        number of active flavors
    nl : int
        number of leptons partecipating to alphaem running
    L : float
        logarithmic ratio of factorization and renormalization scale

    Returns
    -------
    gamma : numpy.ndarray
        adjusted anomalous dimensions
    """
    # if alphaem is fixed then only alphas is varied so gamma[0,1] and gamma[0,2]
    # don't get a variation while gamma[1,1] gets a variation that is O(as2aem1)
    # that we are neglecting
    gamma[1:, 0] = gamma_variation(gamma[1:, 0], order, nf, L)
    if alphaem_running:
        if order[1] >= 2:
            beta0qed = beta.beta_qed((0, 2), nf, nl)
            gamma[0, 2] += beta0qed * gamma[0, 1] * L
        return gamma
