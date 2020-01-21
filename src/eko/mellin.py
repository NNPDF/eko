# -*- coding: utf-8 -*-
"""
    This file contains the implementations of the Mellin transformation and its inverse
    transformation.

    It contains the actual transformations itself, as well as the necessary tools
    such as the definition of paths.
"""

import numpy as np
import numba as nb
import scipy.integrate as integrate

from eko import t_float, t_complex


def compile_integrand(iker, path, jac, do_numba=True):
    """
        Prepares the integration kernel `iker` to be integrated by the
        inverse mellin transform wrapper.

        Parameters
        ----------
            iker : function
                Integration kernel including x^(-N)
            path : function
                Integration path as a function :math:`p(t) : [0,1] \\to \\mathcal C : t \\to p(t)`
            jac : function
                Jacobian of integration path :math:`j(t) = \\frac{dp(t)}{dt}`
            do_numba: bool
                Boolean flag to return a numba compiled function (default: true)
    """

    def integrand(u, extra_args):
        N = path(u)
        prefactor = np.complex(0.0, -0.5 / np.pi)
        result = 2.0 * np.real(prefactor * iker(N, extra_args) * jac(u))
        return result

    if do_numba:
        return nb.njit(integrand)
    else:
        return integrand


def inverse_mellin_transform(integrand, cut, extra_args=(), eps=1e-12):
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
            cut : t_float
                Numeric cut-off parameter to the integration, the actual integration borders are
                determied by :math:`t\\in [c : 1-c]`
            extra_args: any
                Extra arguments to be passed to the integrand beyond the integration variable
            eps: t_float
                Error tolerance (relative and absolute) of the integration


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
        epsabs=eps,
        epsrel=eps,
        limit=LIMIT,
        full_output=1,
    )
    return result[:2]


def get_path_Talbot(r: t_float = 1.0, o: t_float = 0.0):
    """
        Talbot path.

        .. math::
            p_{\\text{Talbot}}(t) =  \\theta \\cdot \\cot(\\theta) + i\\theta, \\theta = \\pi(2t-1)

        Parameters
        ----------
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
    def path(t):
        theta = np.pi * (2.0 * t - 1.0)
        re = 0.0
        if t == 0.5:  # treat singular point seperately
            re = 1.0
        else:
            re = theta / np.tan(theta)
        im = theta
        return o + r * t_complex(np.complex(re, im))

    @nb.njit
    def jac(t):
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


def get_path_line(m: t_float, c: t_float = 1.0):
    """
        Textbook path, i.e. a straight line parallel to the imaginary axis.

        .. math::
            p_{\\text{line}}(t) = c + m \\cdot (2t - 1)

        Parameters
        ----------
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
    def path(t):
        return t_complex(np.complex(c, m * (2 * t - 1)))

    @nb.njit
    def jac(j):  # pylint: disable=unused-argument
        return t_complex(np.complex(0, m * 2))

    return path, jac


def get_path_edge(m: t_float, c: t_float = 1.0, phi: t_complex = np.pi * 2.0 / 3.0):
    """
        Edged path with a given angle.

        .. math::
            p_{\\text{edge}}(t) = c + m\\left|t - \\frac 1 2\\right|\\exp(i\\phi)

        Parameters
        ----------
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
    def path(t):
        if t < 0.5:  # turning point: path is not differentiable in this point
            return c + (0.5 - t) * m * np.exp(np.complex(0, -phi))
        else:
            return c + (t - 0.5) * m * np.exp(np.complex(0, +phi))

    @nb.njit
    def jac(t):
        if t < 0.5:  # turning point: jacobian is not continuous here
            return -m * np.exp(np.complex(0, -phi))
        else:
            return +m * np.exp(np.complex(0, phi))

    return path, jac


def get_path_Cauchy_tan(g: t_float = 1.0, re_offset=0.0):
    """
        Cauchy-distribution like path, extended with tan to infinity.

        .. math::
            p_{\\text{Cauchy}}(t) = \\frac{\\gamma}{u^2 + \\gamma^2} + i u, u = \\tan(\\pi(2t-1)/2)

        Parameters
        ----------
        g : t_float
            intersection of path with real axis

        Returns
        -------
        path : function
            Cauchy path
        jac : function
            derivative of Cauchy path
    """

    @nb.njit
    def path(t):
        u = np.tan(np.pi * (2.0 * t - 1.0) / 2.0)
        re = g / (u * u + g * g)
        return np.complex(re_offset + re, u)

    @nb.njit
    def jac(t):
        arg = np.pi * (2.0 * t - 1.0) / 2.0
        u = np.tan(arg)
        dre = -2.0 * g * u / (u * u + g * g) ** 2
        du = np.pi / np.cos(arg) ** 2
        return np.complex(dre * du, du)

    return path, jac


# TODO if we keep this function open, we might also think about an implementation (t_float)
def mellin_transform(f, N: t_complex):
    """
        Mellin transformation

        Parameters
        ----------
        f : function
            integration kernel :math:`f(x)`
        N : t_complex
            transformation point

        Returns
        -------
        res : t_complex
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
