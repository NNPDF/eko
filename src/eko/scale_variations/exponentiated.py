r"""Contains the scale variation for ``ModSV=exponentiated``."""
import numba as nb

from .. import beta

# Mathematica snippet to compute the gamma variation
# (* define beta fnc - note: convention is different from AB, but matches with EKO *)
# \[Beta]f[k_][a_]:=Sum[-\[Beta][n]*a^(2+n),{n,0,k}]
# (* spell out the all order expansion with taylor + diff eq. *)
# aExpanded[m_,k_]:=Module[{v},
# v=Sum[1/n!t^n (D[a[t],{t,n}]/.{t->0}),{n,0,m}];
# v=v//.{Derivative[n_][a][u_]:>(D[\[Beta]f[k][a[t]],{t,n-1}]/.{t->u})};
# Collect[v,{a0,t},FullSimplify]
# ]
# (* We need to invert the series as we express everything at the modified scale *)
# Module[{s,ss,gg},
# (* get alphas series *)
# s=Series[aExpanded[4,4],{a[0],0,4+1}];
# (*Print@s;*)
# (* we can use this to get the gamma series. *)
# ss=s/.{t->-tt};
# gg=Series[Sum[(ss)^k  g[k-1],{k,1,4}],{a[0],0,4}];
# Print@Collect[gg,{a[0],tt}];
# ]


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
def gamma_variation_qed(gamma, order, nf, L, alphaem_running):
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
    gamma : numpy.ndarray
        adjusted anomalous dimensions
    """
    # if alphaem is fixed then only alphas is varied so gamma[0,1] and gamma[0,2]
    # don't get a variation while gamma[1,1] gets a variation that is O(as2aem1)
    # that we are neglecting
    gamma[1:, 0] = gamma_variation(gamma[1:, 0], order, nf, L)
    if alphaem_running:
        if order[1] >= 2:
            beta0qed = beta.beta_qed((0, 2), nf)
            gamma[0, 2] += beta0qed * gamma[0, 1] * L
        return gamma
