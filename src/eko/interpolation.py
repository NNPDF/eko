# -*- coding: utf-8 -*-
"""
    Library providing all necessary tools for PDF interpolation.

    This library provides a number of functions for generating grids
    as `numpy` arrays:

    * `get_xgrid_linear_at_id`
    * `get_xgrid_linear_at_log`


    This library also provides a class to generate an interpolator `InterpolatorDispatcher`.
    Upon construction the dispatcher generates a number of functions
    to evaluate the interpolator.
"""
import math
import numpy as np
import numba as nb
from eko import t_float

#### Grid generation functions
def get_xgrid_linear_at_id(
    grid_size: int, xmin: t_float = 0.0, xmax: t_float = 1.0, **kwargs
):
    """
    Computes a linear grid on x, maps to `numpy.linspace`
    corresponds to the flag `linear@id`

    Parameters
    ----------
      grid_size : int
        The total size of the grid.
      xmin : t_float
        The minimum x value. Default is 0.
      xmax : t_float
        The maximum x value. Default is 1.

    Returns
    -------
      xgrid : array
        List of grid points in x-space
    """
    return np.linspace(xmin, xmax, grid_size, **kwargs)


def get_xgrid_linear_at_log(
    grid_size: int, xmin: t_float, xmax: t_float = 1.0, **kwargs
):
    """
    Computes a linear grid on log(x), maps to `numpy.logspace`
    corresponds to the flag `linear@log`

    Parameters
    ----------
      grid_size : int
        The total size of the grid.
      xmin : t_float
        The minimum x value.
      xmax : t_float
        The maximum x value. Default is 1.

    Returns
    -------
      xgrid : array
        List of grid points in x-space
    """
    return np.logspace(np.log10(xmin), np.log10(xmax), num=grid_size, **kwargs)

def generate_xgrid(xgrid_type = "log", xgrid_size = 10, xgrid_min = 1e-7, xgrid = None):
    """
        Generates input xgrid

        Parameters
        ----------
            xgrid_type : str
                choose between the different grid generations implemented
            xgrid_size : int
                size of the grid to be generated
            xgrid_min : float
                minimum of the grid to be generated
            xgrid : array
                if `xgrid_type` == `custom`, a `xgrid` must be provided
                which will be outputed after checking for uniqueness

        Returns
        -------
            xgrid : array
                input grid
    """
    if xgrid_type.lower() == "log":
        xgrid = get_xgrid_linear_at_log(xgrid_size, xgrid_min)
    elif xgrid_type.lower() == "linear":
        xgrid = get_xgrid_linear_at_id(xgrid_size, xgrid_min)
    elif xgrid_type.lower() == "custom":
        # if the grid given is custom, it means it comes in the input, but check to be sure
        if xgrid is None:
            raise ValueError(f"xgrid_type {xgrid_type} was chosen, but no xgrid was given")
        # check for uniqueness
        unique_xgrid = np.unique(xgrid)
        if not len(unique_xgrid) == len(xgrid):
            raise ValueError(f"The given grid is not unique: {xgrid}")
        xgrid = unique_xgrid
    else:
        raise NotImplementedError(f"xgrid_type {xgrid_type} not implemented")
    return xgrid

#### Interpolation
class Area:
    """
        Class that define each of the area
        of each of the subgrid interpolators

        Upon construction an array of coefficients
        is generated

        Parameters
        ----------
            lower_index: int
                lower index of the area
            polynom_number: int
                degree of the interpolation polynomial
            block: tuple(int, int)
                kmin and kmax
            xgrid: array
            Grid in x-space from which the interpolators are constructed
    """

    def __init__(self, lower_index, polynom_number, block, xgrid):
        self.poly_number = polynom_number
        self._reference_indices = block
        self.kmin = block[0]
        self.kmax = block[1]
        self.coefs = self.compute_coefs(xgrid)
        self.xmin = xgrid[lower_index]
        self.xmax = xgrid[lower_index + 1]

    @property
    def reference_indices(self):
        return self._reference_indices

    @reference_indices.getter
    def reference_indices(self):
        for k in range(self.kmin, self.kmax + 1):
            if k != self.poly_number:
                yield k

    def compute_coefs(self, xgrid):
        """ Compute the coefficients for this area
        given a grid on x """
        denominator = 1.0
        coeffs = np.ones(1)
        xj = xgrid[self.poly_number]
        for s, k in enumerate(self.reference_indices):
            xk = xgrid[k]
            denominator *= xj - xk
            Mxk_coeffs = -xk * coeffs
            coeffs = np.concatenate(([0.0], coeffs))
            coeffs[: s + +1] += Mxk_coeffs
        coeffs /= denominator
        return coeffs

    def __iter__(self):
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
            xgrid_in : array
                Grid in x-space from which the interpolators are constructed
            polynom_rank : int
                degree of the interpolation polynomial
            list_of_blocks: list(tuple(int, int))
                list of tuples with the (kmin, kmax) values for each area
            is_log_mode: bool (default: True)
            is_mode_N: bool (default: True)
                if true compiles the function on N, otherwise compiles x
            numba_it: bool (default: True)
                if true, the functions are passed through `numba.njit`
    """

    def __init__(
        self,
        xgrid,
        poly_number,
        list_of_blocks,
        is_log_mode=True,
        is_log_inv_mode=False,
        is_mode_N=True,
        numba_it=True,
    ):
        self.poly_number = poly_number
        self.areas = []
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
        if is_mode_N:
            self.compile_N(is_log_mode, is_log_inv_mode)
        else:
            self.compile_X(is_log_mode, is_log_inv_mode)

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
                    highest area <= x?
        """
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

    def compile_X(self, is_log_mode, is_log_inv_mode):
        """
            Compiles the function to evaluate the interpolator in x space.

            .. math::
                P_j(x)

            Parameters
            ----------
                is_log_mode : bool
                    use logarithmic mode?
                is_log_inv_mode : bool
                    use logarithmic inverse mode?
        """

        area_list = self.areas_to_const()

        def evaluate_x(x):
            """
                Get a single Lagrange interpolator in x-space
            """
            res = 0.0
            for j, (xmin, xmax, coefs) in enumerate(area_list):
                if xmin < x <= xmax or (j == 0 and x == xmin):
                    for i, coef in enumerate(coefs):
                        res += coef * pow(x, i)
                    return res
            return res
        # cast to reuse
        eval_x = self.njit(evaluate_x)

        def evaluate_log_x(x):
            """
                Get a single Lagrange interpolator in logarithmic x-space
            """
            return eval_x(np.log(x))

        def evaluate_log_inv_x(x):
            """
                Get a single Lagrange interpolator in logarithmic inverse x-space
            """
            xx = np.log(1.0/x)
            res = 0.0
            for j, (xmin, xmax, coefs) in enumerate(area_list):
                # the ordering is inversed here so, we can't use evaluate_x
                if xmin > xx >= xmax or (j == 0 and xx == xmin):
                    for i, coef in enumerate(coefs):
                        res += coef * pow(xx, i)
                    return res
            return res

        if is_log_mode:
            if is_log_inv_mode:
                self.callable = self.njit(evaluate_log_inv_x)
            else:
                self.callable = self.njit(evaluate_log_x)
        else:
            self.callable = self.njit(evaluate_x)

    def compile_N(self, is_log_mode, is_log_inv_mode):
        """
            Compiles the function to evaluate the interpolator in N space.

            Generates a function `evaluate_Nx` with a (N, x) signature `evaluate_Nx(N, logx)`:

            .. math::
                \\tilde P(N)*exp(- N * log(x))

            The polynomials contain naturally factors of :math:`exp(N * j * log(x_{min/max}))`
            which can be joined with the Mellin inversion factor.

            Parameters
            ----------
                is_log_mode : bool
                    use logarithmic mode?
                is_log_inv_mode : bool
                    use logarithmic inverse mode?
        """
        # compile areas
        area_list = self.areas_to_const()

        def evaluate_log_Nx(N, logx):
            """Get a single Lagrange interpolator in N-space multiplied
            by the Mellin-inversion factor. """
            res = 0.0
            global_coef = 1#np.exp(-N * logx)
            # skip polynom?
            #if logx >= area_list[-1][1]:
            #    return 0.0
            for logxmin, logxmax, coefs in area_list:
                # skip area?
                #if logx >= logxmax:
                #    continue
                umax = N * logxmax
                umin = N * logxmin
                emax = np.exp(N*(logxmax - logx))#umax)
                emin = np.exp(N*(logxmin - logx))#umin)
                for i, coef in enumerate(coefs):
                    tmp = 0.0
                    facti = math.gamma(i + 1) * pow(-1, i) / pow(N, i + 1)
                    for k in range(i + 1):
                        factk = 1.0 / math.gamma(k + 1)
                        pmax = pow(-umax, k) * emax
                        pmin = pow(-umin, k) * emin
                        tmp += factk * (pmax - pmin)
                    res += coef * facti * tmp
            return res * global_coef

        def evaluate_Nx(N, logx):
            """Get a single Lagrange interpolator in N-space multiplied
            by the Mellin-inversion factor. """
            res = 0.0
            for xmin, xmax, coefs in area_list:
                lnxmax = np.log(xmax)
                for i, coef in enumerate(coefs):
                    if xmin == 0.0:
                        low = 0.0
                    else:
                        lnxmin = np.log(xmin)
                        low = np.exp(N * (lnxmin - logx) + i * lnxmin)
                    up = np.exp(N * (lnxmax - logx) + i * lnxmax)
                    res += coef * (up - low) / (N + i)
            return res

        def evaluate_log_inv_Nx(N, logx):
            """Get a single Lagrange interpolator in N-space multiplied
            by the Mellin-inversion factor. """
            loginvx = -logx
            res = 0.0
            global_coef = 1#np.exp(-N * logx)
            # skip polynom?
            #if logx >= area_list[-1][1]:
            #    return 0.0
            for loginvxmin, loginvxmax, coefs in area_list:
                # skip area?
                #if logx >= logxmax:
                #    continue
                umax = N * loginvxmax
                umin = N * loginvxmin
                emax = np.exp(N*(loginvx - loginvxmax))#umax)
                emin = np.exp(N*(loginvx - loginvxmin))#umin)
                for i, coef in enumerate(coefs):
                    tmp = 0.0
                    facti = math.gamma(i + 1) / pow(N, i + 1)
                    for k in range(i + 1):
                        factk = 1.0 / math.gamma(k + 1)
                        pmax = pow(umax, k) * emax
                        pmin = pow(umin, k) * emin
                        tmp += factk * (pmax - pmin)
                    res += coef * facti * tmp
            return res * global_coef

        # compile and set function
        if is_log_mode:
            if is_log_inv_mode:
                self.callable = self.njit(evaluate_log_inv_Nx)
            else:
                self.callable = self.njit(evaluate_log_Nx)
        else:
            self.callable = self.njit(evaluate_Nx)

    def njit(self, function):
        """
            Compile function to Numba if necessary

            Parameters
            ----------
                function : function
                    input function

            Returns
            -------
                function : function
                    evetually compiled output function
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

    def evaluate_log_Nx_2(self, N, logx):
        res = []
        global_coef = 1#np.exp(-N * logx)
        # skip polynom?
        #if logx >= area_list[-1][1]:
        #    return 0.0
        for xmin, xmax, coefs in self.areas_to_const():
            #if logx >= xmax:
            #    continue
            umax = N * xmax
            umin = N * xmin
            emax = np.exp(N*(xmax - logx))#umax)
            emin = np.exp(N*(xmin - logx))#umin)
            for i, coef in enumerate(coefs):
                tmp = []
                facti = math.gamma(i + 1) * pow(-1, i) / pow(N, i + 1)
                for k in range(i + 1):
                    factk = 1.0 / math.gamma(k + 1)
                    pmax = pow(-umax, k) * emax
                    pmin = pow(-umin, k) * emin
                    tmp +=[ factk * pmax, -  factk * pmin]
                res += [(coef, facti, tmp)]
        return res * global_coef

    def evaluate_log_inv_Nx_2(self, N, logx):
        loginvx = -logx
        res = []
        global_coef = 1#np.exp(-N * logx)
        # skip polynom?
        #if logx >= area_list[-1][1]:
        #    return 0.0
        for loginvxmin, loginvxmax, coefs in self.areas_to_const():
            # skip area?
            #if logx >= logxmax:
            #    continue
            umax = N * loginvxmax
            umin = N * loginvxmin
            emax = np.exp(N*(loginvx - loginvxmax))#umax)
            emin = np.exp(N*(loginvx - loginvxmin))#umin)
            for i, coef in enumerate(coefs):
                tmp = []
                facti = math.gamma(i + 1) / pow(N, i + 1)
                for k in range(i + 1):
                    factk = 1.0 / math.gamma(k + 1)
                    pmax = pow(umax, k) * emax
                    pmin = pow(umin, k) * emin
                    tmp += [factk * pmax,  - factk * pmin]
                res += [(coef, facti, tmp)]
        return res * global_coef

class InterpolatorDispatcher:
    """
        Setups the interpolators.

        Upon construction will generate a list of `BasisFunction` objects.
        Each of these `BasisFunction` objects exponses a `callable`
        method (also accessible as the `__call__` method of the class)
        which will be numba-compiled.


        Parameters
        ----------
            xgrid : array
                Grid in x-space from which the interpolators are constructed
            polynom_degree : int
                degree of the interpolation polynomial
            log: bool (default: True)
                use logarithmic interpolation?
            is_log_inv_mode: bool (default: False)
                use logarithmic inverse interpolation?
            mode_N: bool (default: True)
                if true compiles the function on N, otherwise compiles x
    """

    def __init__(self, xgrid, polynomial_degree, log=True, is_log_inv_mode=False, mode_N=True, ):

        xgrid_size = len(xgrid)

        if xgrid_size != len(np.unique(xgrid)):
            raise ValueError(f"xgrid is not unique: {xgrid}")
        if xgrid_size < 2:
            raise ValueError(f"xgrid needs at least 2 points, received {xgrid_size}")
        if polynomial_degree < 1:
            raise ValueError(f"need at least polynomial_degree 1, received {polynomial_degree}")
        if xgrid_size < polynomial_degree:
            raise ValueError(
                f"to interpolate with degree {polynomial_degree} we need at least that much points"
            )

        if log:
            if is_log_inv_mode:
                xgrid = np.log(1.0/xgrid)
            else:
                xgrid = np.log(xgrid)

        # Save the different variables
        self.xgrid = xgrid
        self.polynomial_degree = polynomial_degree
        self.log = log

        # Create blocks
        list_of_blocks = []
        po2 = polynomial_degree // 2
        for i in range(xgrid_size - 1):
            kmin = max(0, i - po2)
            kmax = kmin + polynomial_degree
            if kmax >= xgrid_size:
                kmax = xgrid_size - 1
                kmin = kmax - polynomial_degree
            list_of_blocks.append((kmin, kmax))

        # Generate the basis functions
        basis_functions = []
        for i in range(xgrid_size):
            new_basis = BasisFunction(
                xgrid, i, list_of_blocks,
                is_log_mode=log, is_log_inv_mode=is_log_inv_mode, is_mode_N=mode_N,
            )
            basis_functions.append(new_basis)
        self.basis = basis_functions

    def __iter__(self):
        for basis in self.basis:
            yield basis

    def __getitem__(self, item):
        return self.basis[item]
