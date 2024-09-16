r"""Tools for Mellin inversion.

We provide all necessary toold to deal with the
`inverse Mellin transformation <https://en.wikipedia.org/wiki/Mellin_inversion_theorem>`_.

Although this module provides three different path implementations in practice
only the Talbot path :cite:`Abate`

.. math::
    p_{\text{Talbot}}(t) =  o + r \cdot ( \theta \cot(\theta) + i\theta)\quad
        \text{with}~\theta = \pi(2t-1)

is used, as it results in the most efficient convergence. The default values
for the parameters :math:`r,o` are given by :math:`r = 1/2, o = 0` for
the non-singlet integrals and by :math:`r = \frac{2}{5} \frac{16}{1 - \ln(x)}, o = 1`
for the singlet sector. Note that the non-singlet kernels evolve poles only up to
:math:`N=0` whereas the singlet kernels have poles up to :math:`N=1`.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def Talbot_path(t, r, o):
    r"""Compute Talbot path.

    .. math::
        p_{\text{Talbot}}(t) =  o + r \cdot ( \theta \cot(\theta) + i\theta ),
        \theta = \pi(2t-1)

    Parameters
    ----------
    t : float
        way parameter
    r : float
        scaling parameter - effectively corresponds to the intersection of the path with the
        real axis
    o : float
        offset on real axis

    Returns
    -------
    path : complex
        Talbot path
    """
    theta = np.pi * (2.0 * t - 1.0)
    re = 0.0
    if t == 0.5:  # treat singular point separately
        re = 1.0
    else:
        re = theta / np.tan(theta)
    im = theta
    return o + r * complex(re, im)


@nb.njit(cache=True)
def Talbot_jac(t, r, _o):
    r"""Compute Derivative of Talbot path.

    .. math::
        \frac{dp_{\text{Talbot}}(t)}{dt}

    Parameters
    ----------
    t : float
        way parameter
    r : float
        scaling parameter - effectively corresponds to the intersection of the path with the
        real axis
    o : float
        offset on real axis

    Returns
    -------
    jac : complex
        derivative of Talbot path
    """
    theta = np.pi * (2.0 * t - 1.0)
    re = 0.0
    if t == 0.5:  # treat singular point separately
        re = 0.0
    else:
        re = 1.0 / np.tan(theta)
        re -= theta / (np.sin(theta)) ** 2
    im = 1.0
    return r * np.pi * 2.0 * complex(re, im)


@nb.njit(cache=True)
def line_path(t, m, c):
    r"""Compute textbook path, i.e. a straight line parallel to the imaginary
    axis.

    .. math::
        p_{\text{line}}(t) = c + m \cdot (2t - 1)

    Parameters
    ----------
    t : float
        way parameter
    m : float
        scaling parameter
    c : float
        offset on real axis

    Returns
    -------
    path : complex
        Textbook path
    """
    return complex(c, m * (2 * t - 1))


@nb.njit(cache=True)
def line_jac(_t, m, _c):
    r"""Compute derivative of Textbook path.

    .. math::
        \frac{dp_{\text{line}}(t)}{dt}

    Parameters
    ----------
    t : float
        way parameter
    m : float
        scaling parameter
    o : float
        offset on real axis

    Returns
    -------
    jac : complex
        derivative of Textbook path
    """
    return complex(0, m * 2)


@nb.njit(cache=True)
def edge_path(t, m, c, phi):
    r"""Compute edged path with a given angle.

    .. math::
        p_{\text{edge}}(t) = c + m\left|t - \frac 1 2\right|\exp(i\phi)

    Parameters
    ----------
    t : float
        way parameter
    m : float
        length of the path
    c : float, optional
        intersection of path with real axis
    phi : complex, optional
        bended angle

    Returns
    -------
    path : complex
        Edged path
    """
    if t < 0.5:  # turning point: path is not differentiable in this point
        return c + (0.5 - t) * m * np.exp(complex(0, -phi))
    return c + (t - 0.5) * m * np.exp(complex(0, +phi))


@nb.njit(cache=True)
def edge_jac(t, m, _c, phi):
    r"""Compute derivative of edged path.

    .. math::
        \frac{dp_{\text{edge}}(t)}{dt}

    Parameters
    ----------
    t : float
        way parameter
    m : float
        length of the path
    c : float, optional
        intersection of path with real axis
    phi : complex, optional
        bended angle

    Returns
    -------
    path : complex
        Derivative of edged path
    """
    if t < 0.5:  # turning point: jacobian is not continuous here
        return -m * np.exp(complex(0, -phi))
    return +m * np.exp(complex(0, phi))


spec = [
    ("t", nb.float64),
    ("r", nb.float64),
    ("o", nb.int8),
]


@nb.experimental.jitclass(spec)
class Path:
    """Mellin path dispatcher.

    Parameters
    ----------
    t : float
        way parameter
    logx : float
        Mellin inversion point
    axis_offset: bool
        add offset on the real axis
    """

    def __init__(self, t, logx, axis_offset):
        self.t = t
        # The prescription suggested by :cite:`Abate` for r is 0.4 * M / ( - logx)
        # with M the number of accurate digits; Maria Ubiali suggested in her thesis M = 16.
        # However, this seems to yield unstable results for the OME in the large x region
        # so we add an additional regularization, which makes the path less "edgy" there
        self.r = 0.4 * 16.0 / (0.1 - logx)
        if axis_offset:
            self.o = 1.0
        else:
            self.o = 0.0

    # TODO: make also the other 2 paths available ??
    @property
    def n(self):
        """Return the Mellin moment N."""
        return Talbot_path(self.t, self.r, self.o)

    @property
    def jac(self):
        """Return the Jacobian of the Mellin path."""
        return Talbot_jac(self.t, self.r, self.o)

    @property
    def prefactor(self):
        r"""Return the mellin inversion prefactor :math:`-\frac{i}{\pi}`."""
        return complex(0.0, -1.0 / np.pi)
