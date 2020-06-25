# -*- coding: utf-8 -*-
r"""
    This module contains the implementation of the
    `inverse Mellin transformation <https://en.wikipedia.org/wiki/Mellin_inversion_theorem>`_.

    It contains the actual transformations itself, as well as the necessary tools
    such as the definition of paths.

    The integral routine is provided by `scipy.integrate.quad <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html?highlight=quad#scipy.integrate.quad>`_

    Integration Paths
    -----------------

    Although this module provides four different path implementations (:meth:`get_path_Talbot`,
    :meth:`get_path_line`, :meth:`get_path_edge`, :meth:`get_path_Cauchy_tan`) in practice
    only the Talbot path :cite:`Abate`

    .. math::
        p_{\text{Talbot}}(t) =  o + r \cdot ( \theta \cot(\theta) + i\theta)\quad
            \text{with}~\theta = \pi(2t-1)

    is used, as it results in the most efficient convergence. The default values
    for the parameters :math:`r,o` are given by :math:`r = 1/2, o = 0` for
    the non-singlet integrals and by :math:`r = \frac{2}{5} \frac{16}{1 - \ln(x)}, o = 1`
    for the singlet sector. Note that the non-singlet kernels evolve poles only up to
    :math:`N=0` whereas the singlet kernels have poles up to :math:`N=1`.

"""  # pylint:disable=line-too-long

import numpy as np
import scipy.integrate as integrate
import numba as nb

from eko import t_complex


def compile_integrand(iker, path, jac, do_numba=True):
    r"""
        Prepares the integration kernel `iker` to be integrated by the
        inverse mellin transform wrapper.

        It adds the correct prefactor, resolves the path, adds the jacobian
        and applies the real part.

        Parameters
        ----------
            iker : function
                Integration kernel including :math:`x^(-N)`
            path : function
                Integration path as a function
                :math:`p : [0:1] \to \mathbb C : t \to p(t)`
            jac : function
                Jacobian of integration path :math:`j(t) = \frac{dp(t)}{dt}`
            do_numba: bool
                Boolean flag to return a numba compiled function (default: true)
    """

    def integrand(u, extra_args):
        # Make the extra arguments explicit
        logx = extra_args[0]
        delta_t = extra_args[1]
        path_param = extra_args[2:]
        N = path(u, path_param)
        prefactor = np.complex(0.0, -1.0 / np.pi)
        result = np.real(prefactor * iker(N, logx, delta_t) * jac(u, path_param))
        return result

    if do_numba:
        return nb.njit(integrand)
    else:
        return integrand


def inverse_mellin_transform(integrand, cut, extra_args, epsabs=1e-12, epsrel=1e-8):
    """
        Inverse Mellin transformation.

        Note that the inversion factor :math:`x^{-N}` has already to be *included* in f(N).
        This convention usually improves the convergence of the integral. Typical kernels
        will naturally develop similar factors to which the conversion factor can
        be joined.

        Parameters
        ----------
            integrand: function
                Integrand to be passed to the integration routine.
                The integrand can be generated with the `compile_integrand` function.
            cut : float
                Numeric cut-off parameter to the integration, the actual integration borders are
                determied by :math:`t\\in [c : 1-c]`
            extra_args: any
                Extra arguments to be passed to the integrand beyond the integration variable
            epsabs: float
                absolute error tolerance of the integration
            epsrel: float
                relative error tolerance of the integration


        Returns
        -------
            res : float
                computed point
    """
    LIMIT = 100
    result = integrate.quad(
        integrand,
        0.5,
        1.0 - cut,
        args=extra_args,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=LIMIT,
        full_output=1,
    )
    # if len(result) > 3:
    #    print(result)
    return result[:2]


def get_path_Talbot():
    """
        Talbot path.

        .. math::
            p_{\\text{Talbot}}(t) =  o + r \\cdot ( \\theta \\cot(\\theta) + i\\theta ),
            \\theta = \\pi(2t-1)

        Returns the path and its derivative which then have to be called with the arguments
        listed under `Other Parameters`.

        Other Parameters
        -----------------
            r : t_float
                scaling parameter - effectivly corresponds to the intersection of the path with the
                real axis
            o : t_float
                offset on real axis

        Returns
        -------
            path : function
                Talbot path function
            jac : function
                derivative of Talbot path
    """

    @nb.njit
    def path(t, extra_args):
        r, o = extra_args
        theta = np.pi * (2.0 * t - 1.0)
        re = 0.0
        if t == 0.5:  # treat singular point seperately
            re = 1.0
        else:
            re = theta / np.tan(theta)
        im = theta
        return o + r * t_complex(np.complex(re, im))

    @nb.njit
    def jac(t, extra_args):
        r, o = extra_args  # pylint: disable=unused-variable
        theta = np.pi * (2.0 * t - 1.0)
        re = 0.0
        if t == 0.5:  # treat singular point seperately
            re = 0.0
        else:
            re = 1.0 / np.tan(theta)
            re -= theta / (np.sin(theta)) ** 2
        im = 1.0
        return r * np.pi * 2.0 * t_complex(np.complex(re, im))

    return path, jac


def get_path_line():
    """
        Textbook path, i.e. a straight line parallel to the imaginary axis.

        .. math::
            p_{\\text{line}}(t) = c + m \\cdot (2t - 1)

        Returns the path and its derivative which then have to be called with the arguments
        listed under `Other Parameters`.

        Other Parameters
        ----------------
            m : t_float
                half length of the path
            c : t_float
                intersection of path with real axis

        Returns
        -------
            path : function
                textbook path
            jac : function
                derivative of textbook path
    """

    @nb.njit
    def path(t, extra_args):
        m, c = extra_args
        return t_complex(np.complex(c, m * (2 * t - 1)))

    @nb.njit
    def jac(j, extra_args):  # pylint: disable=unused-argument
        m, c = extra_args  # pylint: disable=unused-variable
        return t_complex(np.complex(0, m * 2))

    return path, jac


def get_path_edge():
    """
        Edged path with a given angle.

        .. math::
            p_{\\text{edge}}(t) = c + m\\left|t - \\frac 1 2\\right|\\exp(i\\phi)

        Returns the path and its derivative which then have to be called with the arguments
        listed under `Other Parameters`.

        Other Parameters
        ----------------
            m : t_float
                length of the path
            c : t_float, optional
                intersection of path with real axis - defaults to 1
            phi : t_complex, optional
                bended angle - defaults to +135° with respect to positive x axis
        Returns
        -------
            path : function
                Edged path
            jac : function
                derivative of edged path
    """

    @nb.njit
    def path(t, extra_args):
        m, c, phi = extra_args
        if t < 0.5:  # turning point: path is not differentiable in this point
            return c + (0.5 - t) * m * np.exp(np.complex(0, -phi))
        else:
            return c + (t - 0.5) * m * np.exp(np.complex(0, +phi))

    @nb.njit
    def jac(t, extra_args):
        m, c, phi = extra_args  # pylint: disable=unused-variable
        if t < 0.5:  # turning point: jacobian is not continuous here
            return -m * np.exp(np.complex(0, -phi))
        else:
            return +m * np.exp(np.complex(0, phi))

    return path, jac


def get_path_Cauchy_tan():
    """
        Cauchy-distribution like path, extended with tan to infinity.

        .. math::
            p_{\\text{Cauchy}}(t) = \\frac{\\gamma}{u^2 + \\gamma^2} + i u, u = \\tan(\\pi(2t-1)/2)

        Returns the path and its derivative which then have to be called with the arguments
        listed under `Other Parameters`.

        Other Parameters
        ---------------
            gamma : t_float
                intersection of path with real axis

        Returns
        -------
            path : function
                Cauchy path
            jac : function
                derivative of Cauchy path
    """

    @nb.njit
    def path(t, extra_args):
        g, re_offset = extra_args
        u = np.tan(np.pi * (2.0 * t - 1.0) / 2.0)
        re = g / (u * u + g * g)
        return np.complex(re_offset + re, u)

    @nb.njit
    def jac(t, extra_args):
        g, re_offset = extra_args  # pylint: disable=unused-variable
        arg = np.pi * (2.0 * t - 1.0) / 2.0
        u = np.tan(arg)
        dre = -2.0 * g * u / (u * u + g * g) ** 2
        du = np.pi / np.cos(arg) ** 2
        return np.complex(dre * du, du)

    return path, jac


def mellin_transform(f, N: complex):
    """
        Mellin transformation

        Parameters
        ----------
            f : function
                integration kernel :math:`f(x)`
            N : complex
                transformation point

        Returns
        -------
            res : complex
                computed point
    """

    @nb.jit(forceobj=True)  # due to the integration kernel not being necessarily numba
    def integrand(x):
        xToN = pow(x, N - 1) * f(x)
        return xToN

    # do real + imaginary part seperately
    r, re = integrate.quad(lambda x: np.real(integrand(x)), 0, 1, full_output=1)[:2]
    i, ie = integrate.quad(lambda x: np.imag(integrand(x)), 0, 1, full_output=1)[:2]
    result = t_complex(complex(r, i))
    error = t_complex(complex(re, ie))
    return result, error
