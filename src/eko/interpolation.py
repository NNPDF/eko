# -*- coding: utf-8 -*-
"""
Library providing all necessary tools for PDF interpolation.

This library also provides a class to generate the interpolator :class:`InterpolatorDispatcher`.
Upon construction the dispatcher generates a number of functions
to evaluate the interpolator.
"""
import logging
import math

import numba as nb
import numpy as np

logger = logging.getLogger(__name__)


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
        xgrid: numpy.ndarray
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
        """Iterate over all indices which are part of the block"""
        for k in range(self.kmin, self.kmax + 1):
            if k != self.poly_number:
                yield k

    def _compute_coefs(self, xgrid):
        """Compute the coefficients for this area given a grid on x"""
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
        """Iterates the generated coefficients"""
        for coef in self.coefs:
            yield coef


@nb.njit("c16(c16,f8,f8[:,:])", cache=True)
def log_evaluate_Nx(N, logx, area_list):
    r"""
    Evaluates a single logarithmic Lagrange interpolator in N-space multiplied
    by the Mellin-inversion factor.

    .. math::
        \tilde p(N)*\exp(- N * \ln(x))

    Parameters
    ----------
        N : complex
            Mellin variable
        logx : float
            logarithm of inversion point
        area_list : list
            area configuration of basis function

    Returns
    -------
        res : float
            kernel * inversion factor
    """
    res = 0.0
    for a in area_list:
        logxmin = a[0]
        logxmax = a[1]
        coefs = a[2:]
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
                # this condition is actually not necessary in python since
                # there pow(0,0) == 1 and apparently this is inherited in
                # Numba/C, however this is mathematically cleaner
                if umax == 0.0 and k == 0:
                    pmax = emax
                else:
                    pmax = pow(-umax, k) * emax
                # drop factor by analytics?
                if logx >= logxmin:
                    pmin = 0
                else:
                    pmin = pow(-umin, k) * emin
                tmp += factk * (pmax - pmin)
            res += coef * facti * tmp
    return res


@nb.njit("c16(c16,f8,f8[:,:])", cache=True)
def evaluate_Nx(N, logx, area_list):
    r"""
    Evaluates a single linear Lagrange interpolator in N-space multiplied
    by the Mellin-inversion factor.

    .. math::
        \tilde p(N)*\exp(- N * \ln(x))

    Parameters
    ----------
        N : complex
            Mellin variable
        logx : float
            logarithm of inversion point
        area_list : list
            area configuration of basis function

    Returns
    -------
        res : float
            basis function * inversion factor
    """
    res = 0.0
    for a in area_list:
        xmin = a[0]
        xmax = a[1]
        coefs = a[2:]
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


# TODO lift to runcard?
_atol_eps = 10 * np.finfo(float).eps


@nb.njit("f8(f8,f8[:,:])", cache=True)
def evaluate_x(x, area_list):
    """
    Get a single linear Lagrange interpolator in x-space

    .. math::
        p(x)

    Parameters
    ----------
        x : float
            interpolation point
        area_list : list
            area configuration of basis function

    Returns
    -------
        res : float
            basis function(x)
    """
    res = 0.0
    for j, a in enumerate(area_list):
        xmin = a[0]
        xmax = a[1]
        coefs = a[2:]
        if xmin < x <= xmax or (j == 0 and np.abs(x - xmin) < _atol_eps):
            for i, coef in enumerate(coefs):
                res += coef * pow(x, i)
            return res

    return res


@nb.njit("f8(f8,f8[:,:])", cache=True)
def log_evaluate_x(x, area_list):
    """
    Get a single logarithmic Lagrange interpolator in x-space

    .. math::
        p(x)

    Parameters
    ----------
        x : float
            interpolation point
        area_list : list
            area configuration of basis function

    Returns
    -------
        res : float
            basis function(x)
    """
    x = np.log(x)
    return evaluate_x(x, area_list)


class BasisFunction:
    """
    Object containing a list of areas for a given polynomial number
    defined by (xmin-xmax) and containing a list of coefficients.

    Upon construction will generate all areas and generate and compile
    a function to evaluate in N (or x) the iterpolator

    Parameters
    ----------
        xgrid : numpy.ndarray
            Grid in x-space from which the interpolators are constructed
        poly_number : int
            number of polynomial
        list_of_blocks: list(tuple(int, int))
            list of tuples with the (kmin, kmax) values for each area
        mode_log: bool
            use logarithmic interpolation?
        mode_N: bool
            if true compiles the function on N, otherwise compiles x
    """

    def __init__(
        self,
        xgrid,
        poly_number,
        list_of_blocks,
        mode_log=True,
        mode_N=True,
    ):
        self.poly_number = poly_number
        self.areas = []
        self._mode_log = mode_log
        self.mode_N = mode_N

        # create areas
        for i, block in enumerate(list_of_blocks):
            if block[0] <= poly_number <= block[1]:
                new_area = Area(i, self.poly_number, block, xgrid)
                self.areas.append(new_area)
        if not self.areas:
            raise ValueError("Error: no areas were generated")
        self.areas_representation = self.areas_to_const()

        # compile
        # TODO move this to InterpolatorDispatcher
        self.callable = None
        if self.mode_N:
            self.compile_n()
        else:
            self.compile_x()

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
        Returns an array containing all areas with
        (`xmin`, `xmax`, `numpy.array` of coefficients)

        Returns
        -------
            numpy.ndarray
                area config
        """
        # This is necessary as numba will ask for everything
        # to be inmutable
        area_list = []
        for area in self:
            area_list.append([area.xmin, area.xmax, *area.coefs])
        return np.array(area_list)

    def compile_x(self):
        """
        Compiles the function to evaluate the interpolator
        in x space
        """
        if self._mode_log:
            self.callable = log_evaluate_x
        else:
            self.callable = evaluate_x

    def evaluate_x(self, x):
        """
        Evaluate basis function in x-space (regardless of the true space).

        Parameters
        ----------
            x : float
                evaluated point

        Returns
        -------
            res : float
                p(x)
        """
        if self.mode_N:
            old_call = self.callable
            self.compile_x()
            res = self.callable(x, self.areas_representation)
            self.callable = old_call
        else:
            res = self.callable(x, self.areas_representation)
        return res

    def compile_n(self):
        r"""
        Compiles the function to evaluate the interpolator in N space.

        Generates a function :meth:`evaluate_Nx` with a (N, logx) signature.

        .. math::
            \tilde p(N)*\exp(- N * \ln(x))

        The polynomials contain naturally factors of :math:`\exp(N * j * \ln(x_{min/max}))`
        which can be joined with the Mellin inversion factor.
        """

        if self._mode_log:
            self.callable = log_evaluate_Nx
        else:
            self.callable = evaluate_Nx

    def __iter__(self):
        for area in self.areas:
            yield area

    def __call__(self, *args, **kwargs):
        args = list(args)
        args.append(self.areas_representation)
        return self.callable(*args, **kwargs)


class InterpolatorDispatcher:
    """
    Setups the interpolator.

    Upon construction will generate a list of :class:`BasisFunction` objects.
    Each of these :class:`BasisFunction` objects exponses a `callable`
    method (also accessible as the `__call__` method of the class)
    which will be numba-compiled.


    Parameters
    ----------
        xgrid_in : numpy.ndarray
            Grid in x-space from which the interpolators are constructed
        polynomial_degree : int
            degree of the interpolation polynomial
        log: bool
            Whether it is a log or linear interpolator
        mode_N: bool
            if true compiles the function on N, otherwise compiles x
    """

    def __init__(self, xgrid, polynomial_degree, log=True, mode_N=True):
        # sanity checks
        xgrid_size = len(xgrid)
        ugrid = np.array(np.unique(xgrid), np.float_)
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
        logger.info(
            "Interpolation: number of points = %d, polynomial degree = %d, logarithmic = %s",
            xgrid_size,
            polynomial_degree,
            log,
        )

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
                xgrid, i, list_of_blocks, mode_log=log, mode_N=mode_N
            )
            basis_functions.append(new_basis)
        self.basis = basis_functions

    @classmethod
    def from_dict(cls, operators_card, mode_N=True):
        """
        Create object from dictionary.

        .. list-table:: operators runcard parameters
            :header-rows: 1

            *   - Name
                - Type
                - default
                - description
            *   - ``interpolation_xgrid``
                - :py:obj:`list(float)`
                - [required]
                - the interpolation grid
            *   - ``interpolation_polynomial_degree``
                - :py:obj:`int`
                - ``4``
                - polynomial degree of the interpolating function
            *   - ``interpolation_is_log``
                - :py:obj:`bool`
                - ``True``
                - use logarithmic interpolation?

        Parameters
        ----------
            operators_card : dict
                input configurations
        """
        # load xgrid
        xgrid = operators_card["interpolation_xgrid"]
        if len(xgrid) == 0:
            raise ValueError("Empty xgrid!")
        if xgrid[0] == "make_grid":
            xgrid = make_grid(*xgrid[1:])
        is_log_interpolation = bool(operators_card["interpolation_is_log"])
        polynom_rank = operators_card["interpolation_polynomial_degree"]
        return cls(
            xgrid,
            polynom_rank,
            log=is_log_interpolation,
            mode_N=mode_N,
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


def make_grid(
    n_low, n_mid, n_high=0, x_min=1e-7, x_low=0.1, x_high=0.9, x_high_max=1.0 - 1e-4
):
    """
    Creates a log-lin-log-spaced grid.

    1.0 is always part of the grid and the final grid is unified, i.e. esp. points that might
    appear twice at the borders of the regions are unified.

    Parameters
    ----------
        n_low : int
            points in the small-x region
        n_mid : int
            points in the mediun-x region
        n_high : int
            points in the large-x region
        x_min : float
            minimum x (included)
        x_low : float
            seperation point between small and medium
        x_high_max : float
            closest point before 1

    Returns
    -------
        xgrid : numpy.ndarray
            generated grid
    """
    # low
    if n_mid == 0:
        x_low = 1.0
    xgrid_low = np.geomspace(x_min, x_low, n_low)
    # high
    if n_high == 0:
        x_high = 1.0
        xgrid_high = np.array([])
    else:
        xgrid_high = 1.0 - np.geomspace(1 - x_high, 1 - x_high_max, n_high)
    # mid
    xgrid_mid = np.linspace(x_low, x_high, n_mid)
    # join
    xgrid = np.unique(np.concatenate((xgrid_low, xgrid_mid, xgrid_high, np.array([1]))))
    return xgrid
