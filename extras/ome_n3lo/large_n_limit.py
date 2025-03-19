"""This file contains the large-N limit of the diagonal Matrix elements.

The expansions are obtained using the notebook Agg_Aqq_largex_expansion.nb.

We note that:
    * the limit of :math:`A_{qq}` is the same for valence-like and singlet-like expansions,
       i.e. the local and singular parts are the same
    * the :math:`A_{qq,ps}` term is vanishing in the large-x limit, i.e. it's only regular.

In the following we make use of a small trick, which is available due to complex continuation.
Although we are interested in the large-N limit, we can obtain that by observing the small-N limit.
It is :math:`S_1(0)=0` (by continuation), :math:`S_1(1)=1` and :math:`S_1(N\to\infty) = \ln(N) - \gamma_E`.

"""
from ekore.harmonics import S1


def Aqq_asymptotic(n, nf):
    """The N3LO quark-to-quark transition matrix element large-N limit."""
    return (
        (20.362519064781296 - 3.4050138869326796 * nf) * S1(n)
        - 51.033843609253296
        + 3.1144841058729096 * nf
    )


def Agg_asymptotic(n, nf):
    """The N3LO gluon-to-gluon transition matrix element large-N limit.
    Follwing :cite:`Ablinger:2022wbb`:
        * the fist part contains the limit of eq. 2.6 (except for :math:`a_{gg}^{(3)}`)
        * the second part comes from eq. 4.6 and 4.7.
    """
    Agg_asy_incomplete = (
        (-669.1554507291286 + 41.84286985333757 * nf) * S1(n)
        - 565.4465327471261
        + 28.65462637880661 * nf
    )
    agg_asy = (
        -49.5041510989361 * (-14.442649813264895 + nf) * S1(n)
        + 619.2420126046355
        - 17.52475977636971 * nf
    )
    return agg_asy + Agg_asy_incomplete
