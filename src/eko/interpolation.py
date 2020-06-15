# -*- coding: utf-8 -*-
"""
    Library providing all necessary tools for PDF interpolation.

    This library also provides a class to generate the interpolator `InterpolatorDispatcher`.
    Upon construction the dispatcher generates a number of functions
    to evaluate the interpolator.
"""
import logging
import math

import numpy as np
import numba as nb

logger = logging.getLogger(__name__)

#### Interpolation
class Area:
    """
        Class that define each of the area
        of each of the subgrid interpolators.

        Upon construction an array of coefficients
        is generated.

        Parameters
        ----------
            lower_index: int
                lower index of the area
            poly_number: int
                number of polynomial
            block: tuple(int, int)
                kmin and kmax
            xgrid: array(float)
                Grid in x-space from which the interpolators are constructed
    """

    def __init__(self, lower_index, poly_number, block, xgrid):
        # check range
        if poly_number < block[0] or block[1] < poly_number:
            raise ValueError(
                f"polynom #{poly_number} cannot be a part of the block which"
                f"spans from {block[0]} to {block[1]}"
            )
        self.xmin = xgrid[lower_index]
        self.xmax = xgrid[lower_index + 1]
        self.poly_number = poly_number
        self.kmin = block[0]
        self.kmax = block[1]
        self.coefs = self._compute_coefs(xgrid)

    def _reference_indices(self):
        """ Iterate over all indices which are part of the block """
        for k in range(self.kmin, self.kmax + 1):
            if k != self.poly_number:
                yield k

    def _compute_coefs(self, xgrid):
        """ Compute the coefficients for this area given a grid on x """
        denominator = 1.0
        coeffs = np.ones(1)
        xj = xgrid[self.poly_number]
        for s, k in enumerate(self._reference_indices()):
            xk = xgrid[k]
            denominator *= xj - xk
            Mxk_coeffs = -xk * coeffs
            coeffs = np.concatenate(([0.0], coeffs))
            coeffs[: s + +1] += Mxk_coeffs
        coeffs /= denominator
        return coeffs

    def __iter__(self):
        """ Iterates the generated coefficients """
        for coef in self.coefs:
            yield coef


class BasisFunction:
    """
        Object containing a list of areas for a given polynomial number
        defined by (xmin-xmax) and containing a list of coefficients.

        Upon construction will generate all areas and generate and compile
        a function to evaluate in N (or x) the iterpolator

        Parameters
        ----------
            xgrid : array
                Grid in x-space from which the interpolators are constructed
            poly_number : int
                number of polynomial
            list_of_blocks: list(tuple(int, int))
                list of tuples with the (kmin, kmax) values for each area
            mode_log: bool (default: True)
                use logarithmic interpolation?
            mode_N: bool (default: True)
                if true compiles the function on N, otherwise compiles x
            numba_it: bool (default: True)
                if true, the functions are passed through `numba.njit`
    """

    def __init__(
        self,
        xgrid,
        poly_number,
        list_of_blocks,
        mode_log=True,
        mode_N=True,
        numba_it=True,
    ):
        self.poly_number = poly_number
        self.areas = []
        self._mode_log = mode_log
        self.mode_N = mode_N
        self.numba_it = numba_it

        # create areas
        for i, block in enumerate(list_of_blocks):
            if block[0] <= poly_number <= block[1]:
                new_area = Area(i, self.poly_number, block, xgrid)
                self.areas.append(new_area)
        if not self.areas:
            raise ValueError("Error: no areas were generated")

        # compile
        self.callable = None
        if self.mode_N:
            self.compile_N()
        else:
            self.compile_X()

    def is_below_x(self, x):
        """
            Are all areas below x?

            Parameters
            ----------
                x : float
                    reference value
            Returns
            --------
                is_below_x : bool
                    xmax of highest area <= x?
        """
        # Log if needed
        if self._mode_log:
            x = np.log(x)
        # note that ordering is important!
        return self.areas[-1].xmax <= x

    def areas_to_const(self):
        """
            Retruns a tuple of tuples, one for each area
            each containing
            (`xmin`, `xmax`, `numpy.array` of coefficients)
        """
        # This is necessary as numba will ask for everything
        # to be inmutable
        area_list = []
        for area in self:
            area_list.append((area.xmin, area.xmax, area.coefs))
        return tuple(area_list)

    def compile_X(self):
        """
            Compiles the function to evaluate the interpolator
            in x space

            .. math::
                p(x)

            Parameters
            ----------
                x : t_float
                    Evaluated point
                conf : dict
                    dictionary of values for the coefficients of the interpolator

            Returns
            -------
                p(x) : t_float
                    Evaluated polynom at x
        """

        area_list = self.areas_to_const()

        def evaluate_x(x):
            """Get a single Lagrange interpolator in x-space  """
            res = 0.0
            for j, (xmin, xmax, coefs) in enumerate(area_list):
                if xmin < x <= xmax or (j == 0 and x == xmin):
                    for i, coef in enumerate(coefs):
                        res += coef * pow(x, i)
                    return res

            return res

        # parse to reuse it
        nb_eval_x = self.njit(evaluate_x)

        def log_evaluate_x(x):
            """Get a single Lagrange interpolator in x-space  """
            return nb_eval_x(np.log(x))

        if self._mode_log:
            self.callable = self.njit(log_evaluate_x)
        else:
            self.callable = self.njit(evaluate_x)

    def evaluate_x(self, x):
        """
            Evaluate basis function in x-space.

            Parameters
            ----------
                x : t_float
                    evaluated point

            Returns
            -------
                res : t_float
                    p(x)
        """
        if self.mode_N:
            old_call = self.callable
            old_numba = self.numba_it
            self.numba_it = False
            self.compile_X()
            res = self.callable(x)
            self.callable = old_call
            self.numba_it = old_numba
        else:
            res = self.callable(x)
        return res

    def compile_N(self):
        """
            Compiles the function to evaluate the interpolator in N space.

            Generates a function `evaluate_Nx` with a (N, x) signature `evaluate_Nx(N, logx)`.

            .. math::
                \\tilde p(N)*exp(- N * log(x))

            The polynomials contain naturally factors of :math:`exp(N * j * log(x_{min/max}))`
            which can be joined with the Mellin inversion factor.

            Parameters
            ----------
                N : t_float
                    Evaluated point in N-space
                logx : t_float
                    Mellin-inversion point :math:`log(x)`

            Returns
            -------
                p(N)*x^{-N} : t_complex
                    Evaluated polynomial at N times x^{-N}
        """
        area_list = self.areas_to_const()

        def log_evaluate_Nx(N, logx):
            """Get a single Lagrange interpolator in N-space multiplied
            by the Mellin-inversion factor. """
            res = 0.0
            for logxmin, logxmax, coefs in area_list:
                # skip area completely?
                if logx >= logxmax:
                    continue
                umax = N * logxmax
                umin = N * logxmin
                emax = np.exp(N * (logxmax - logx))
                emin = np.exp(N * (logxmin - logx))
                for i, coef in enumerate(coefs):
                    tmp = 0.0
                    facti = math.gamma(i + 1) * pow(-1, i) / pow(N, i + 1)
                    for k in range(i + 1):
                        factk = 1.0 / math.gamma(k + 1)
                        pmax = pow(-umax, k) * emax
                        # drop factor by analytics?
                        if logx >= logxmin:
                            pmin = 0
                        else:
                            pmin = pow(-umin, k) * emin
                        tmp += factk * (pmax - pmin)
                    res += coef * facti * tmp
            return res

        def evaluate_Nx(N, logx):
            """Get a single Lagrange interpolator in N-space multiplied
            by the Mellin-inversion factor. """
            res = 0.0
            for xmin, xmax, coefs in area_list:
                lnxmax = np.log(xmax)
                # skip area completely?
                if logx >= lnxmax:
                    continue
                for i, coef in enumerate(coefs):
                    if xmin == 0.0:
                        low = 0.0
                    else:
                        lnxmin = np.log(xmin)
                        low = np.exp(N * (lnxmin - logx) + i * lnxmin)
                    up = np.exp(N * (lnxmax - logx) + i * lnxmax)
                    res += coef * (up - low) / (N + i)
            return res

        if self._mode_log:
            self.callable = self.njit(log_evaluate_Nx)
        else:
            self.callable = self.njit(evaluate_Nx)

    def njit(self, function):
        """
            Compiles the function to Numba, if necessary.

            Parameters
            -----------
                function : function
                    function to compile

            Returns
            -------
                function : function
                    compiled function, if needed
        """
        if self.numba_it:
            return nb.njit(function)
        else:
            return function

    def __iter__(self):
        for area in self.areas:
            yield area

    def __call__(self, *args, **kwargs):
        return self.callable(*args, **kwargs)


class InterpolatorDispatcher:
    """
        Setups the interpolator.

        Upon construction will generate a list of `BasisFunction` objects.
        Each of these `BasisFunction` objects exponses a `callable`
        method (also accessible as the `__call__` method of the class)
        which will be numba-compiled.


        Parameters
        ----------
            xgrid_in : array
                Grid in x-space from which the interpolators are constructed
            polynomial_degree : int
                degree of the interpolation polynomial
            log: bool  (default: True)
                Whether it is a log or linear interpolator
            mode_N: bool (default: True)
                if true compiles the function on N, otherwise compiles x
            numba_it : bool (default: True)
                compile with numba?
    """

    def __init__(self, xgrid, polynomial_degree, log=True, mode_N=True, numba_it=True):
        # sanity checks
        xgrid_size = len(xgrid)
        ugrid = np.unique(xgrid)
        if xgrid_size != len(ugrid):
            raise ValueError(f"xgrid is not unique: {xgrid}")
        xgrid = ugrid
        if xgrid_size < 2:
            raise ValueError(f"xgrid needs at least 2 points, received {xgrid_size}")
        if polynomial_degree < 1:
            raise ValueError(
                f"need at least polynomial_degree 1, received {polynomial_degree}"
            )
        if xgrid_size <= polynomial_degree:
            raise ValueError(
                f"to interpolate with degree {polynomial_degree} "
                " we need at least that much points + 1"
            )
        logger.info("Log interpolation: %s", log)
        # keep a true copy of grid
        self.xgrid_raw = xgrid
        # henceforth xgrid might no longer be the input!
        # which is ok, because for most of the code this is all we need to do
        # to distinguish log and non-log
        if log:
            xgrid = np.log(xgrid)

        # Save the different variables
        self.xgrid = xgrid
        self.polynomial_degree = polynomial_degree
        self.log = log

        # Create blocks
        list_of_blocks = []
        po2 = polynomial_degree // 2
        # if degree is even, we can not split the block symmetric, e.g. deg=2 -> |-|-|
        # so, in case of doubt use the block, which lays higher, i.e.
        # we're not allowed to go so deep -> make po2 smaller
        if polynomial_degree % 2 == 0:
            po2 -= 1
        # iterate areas: there is 1 less then number of points
        for i in range(xgrid_size - 1):
            kmin = max(0, i - po2)
            kmax = kmin + polynomial_degree
            if kmax >= xgrid_size:
                kmax = xgrid_size - 1
                kmin = kmax - polynomial_degree
            b = (kmin, kmax)
            list_of_blocks.append(b)

        # Generate the basis functions
        basis_functions = []
        for i in range(xgrid_size):
            new_basis = BasisFunction(
                xgrid, i, list_of_blocks, mode_log=log, mode_N=mode_N, numba_it=numba_it
            )
            basis_functions.append(new_basis)
        self.basis = basis_functions

    @classmethod
    def from_dict(cls, setup, mode_N=True, numba_it=True):
        """
            Create object from dictionary.

            Read keys:

                - interpolation_xgrid : required, basis grid
                - interpolation_is_log : default=True, use logarithmic interpolation?
                - interpolation_polynomial_degree : default=4, polynomial degree of interpolation

            Parameters
            ----------
                setup : dict
                    input configurations
        """
        xgrid = setup["interpolation_xgrid"]
        is_log_interpolation = bool(setup.get("interpolation_is_log", True))
        polynom_rank = setup.get("interpolation_polynomial_degree", 4)

        # Generate the dispatcher for the basis functions
        return cls(
            xgrid,
            polynom_rank,
            log=is_log_interpolation,
            mode_N=mode_N,
            numba_it=numba_it,
        )

    def __eq__(self, other):
        """Checks equality"""
        checks = [
            len(self.xgrid_raw) == len(other.xgrid_raw),
            self.log == other.log,
            self.polynomial_degree == other.polynomial_degree,
        ]
        # check elements after shape
        return all(checks) and np.allclose(self.xgrid_raw, other.xgrid_raw)

    def __iter__(self):
        # return iter(self.basis)
        for basis in self.basis:
            yield basis

    def __getitem__(self, item):
        return self.basis[item]

    def get_interpolation(self, targetgrid):
        """
            Computes interpolation matrix between `targetgrid` and `xgrid`.

            .. math::
                f(targetgrid) = R \\cdot f(xgrid)

            Parameters
            ----------
                targetgrid : array
                    grid to interpolate to

            Returns
            -------
                R : array
                    interpolation matrix, do be multiplied from the left(!)
        """
        # trivial?
        if len(targetgrid) == len(self.xgrid_raw) and np.allclose(
            targetgrid, self.xgrid_raw
        ):
            return np.eye(len(self.xgrid_raw))
        # compute map
        out = []
        for x in targetgrid:
            l = []
            for b in self.basis:
                l.append(b.evaluate_x(x))
            out.append(l)
        return np.array(out)

    def to_dict(self):
        """
            Returns the configuration for the underlying xgrid (from which the instance can
            be created again).

            The output dictionary contains a numpy array.

            Returns
            -------
                ret : dict
                    full grid configuration
        """
        ret = {
            "interpolation_xgrid": self.xgrid_raw,
            "interpolation_polynomial_degree": self.polynomial_degree,
            "interpolation_is_log": self.log,
        }
        return ret
