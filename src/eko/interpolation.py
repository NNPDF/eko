# -*- coding: utf-8 -*-
"""
    Library providing all necessary tools for PDF interpolation.

    This library provides a number of functions for generating grids
    as `numpy` arrays:

        - `get_xgrid_linear_at_id`
        - `get_xgrid_linear_at_log`

    This library also provides a class to generate the interpolator `InterpolatorDispatcher`.
    Upon construction the dispatcher generates a number of functions
    to evaluate the interpolator.
"""
import math
import numpy as np
import scipy.integrate as integrate
import numba as nb

from eko import t_float
import eko.splitting_functions_LO as sf_LO
import eko.alpha_s as alpha_s

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


def generate_xgrid(
    xgrid_type="log", xgrid_size=10, xgrid_min=1e-7, xgrid=None, **kwargs
):
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
        xgrid = get_xgrid_linear_at_log(xgrid_size, xgrid_min, **kwargs)
    elif xgrid_type.lower() == "linear":
        xgrid = get_xgrid_linear_at_id(xgrid_size, xgrid_min, **kwargs)
    elif xgrid_type.lower() == "custom":
        # if the grid given is custom, it means it comes in the input, but check to be sure
        if xgrid is None:
            raise ValueError(
                f"xgrid_type {xgrid_type} was chosen, but no xgrid was given"
            )
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
        of each of the subgrid interpolators.

        Upon construction an array of coefficients
        is generated.

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
        self.lower_index = lower_index
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
        if mode_N:
            self.compile_N(mode_log)
        else:
            self.compile_X(mode_log)

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

    def compile_X(self, mode_log=True):
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

        if mode_log:
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
        old_call = self.callable
        self.compile_X()
        res = self.callable(x)
        self.callable = old_call
        return res

    def compile_N(self, mode_log=True):
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
                for i, coef in enumerate(coefs):
                    if xmin == 0.0:
                        low = 0.0
                    else:
                        lnxmin = np.log(xmin)
                        low = np.exp(N * (lnxmin - logx) + i * lnxmin)
                    up = np.exp(N * (lnxmax - logx) + i * lnxmax)
                    res += coef * (up - low) / (N + i)
            return res

        if mode_log:
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


class MellinPrimitive:
    """
        Represents a single grid point primitive.

        Parameters
        ----------
            x : t_float
                grid point
            polynomial_degree : int
                degree of interpolation polynomial
    """
    def __init__(self,lnx,polynomial_degree):
        self._lnx = lnx
        self._polynomial_degree = polynomial_degree

    def get_polynomials_generator(self):
        """
            Returns the generator for all polynomials
        """
        lnx = self._lnx
        def p(k,im,re=1):
            N = re + 1j*im
            res = 0
            for l in range(k+1):
                res += np.power(-lnx*N,l)/math.gamma(l+1)
            res *= math.gamma(k+1) / np.power(N,k+1) * np.power(-1,k)
            return res
        return nb.njit(p)

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
    """

    def __init__(self, xgrid, polynomial_degree, log=True, mode_N=True):

        xgrid_size = len(xgrid)

        ugrid = np.unique(xgrid)
        if xgrid_size != len(ugrid):
            raise ValueError(f"xgrid is not unique: {xgrid}")
        xgrid = ugrid
        if xgrid_size < 2:
            raise ValueError(f"xgrid needs at least 2 points, received {xgrid_size}")
        if polynomial_degree < 1:
            raise ValueError(f"need at least polynomial_degree 1, received {polynomial_degree}")
        if xgrid_size < polynomial_degree:
            raise ValueError(
                f"to interpolate with degree {polynomial_degree} we need at least that much points"
            )

        self.xgrid_raw = xgrid
        if log:
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
                xgrid, i, list_of_blocks, mode_log=log, mode_N=mode_N
            )
            basis_functions.append(new_basis)
        self.basis = basis_functions

    def __eq__(self,other):
        checks = [
            len(self.xgrid_raw) == len(other.xgrid_raw),
            self.log == other.log,
            self.polynomial_degree == other.polynomial_degree
        ]
        # check elements after shape
        return all(checks) and np.allclose(self.xgrid_raw, other.xgrid_raw)

    def __iter__(self):
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
        out = []
        for x in targetgrid:
            l = []
            for b in self.basis:
                l.append(b.evaluate_x(x))
            out.append(l)
        return np.array(out)

    def get_ns_ker(self, constants, nf, delta_t,j,k,poly_power):
        """
            Returns non-singlet intergration kernel.
        """
        CA = constants.CA
        CF = constants.CF
        beta_0 = alpha_s.beta_0(nf, CA, CF, constants.TF)
        lnxj = self.xgrid[j]
        lnxk = self.xgrid[k]
        omega = lnxj - lnxk
        raw = MellinPrimitive(lnxj, self.polynomial_degree)
        p = raw.get_polynomials_generator()
        def ker(im, re=1):
            """true non-siglet integration kernel"""
            N = re + 1j*im
            lnE = -delta_t * sf_LO.gamma_ns_0(N, nf, CA, CF) / beta_0
            interpoln = p(poly_power,im,re) * np.exp(re * omega)
            return np.real(np.exp(lnE) * interpoln) / np.pi
        return nb.njit(ker)

    def get_raw_ns(self, constants, nf, delta_t):
        """
            Computes raw operator.
        """
        raw = []
        raw_errors = []
        for j,lnxj in enumerate(self.xgrid):
            line = []
            line_err = []
            for k,lnxk in enumerate(self.xgrid):
                elem = []
                elem_err = []
                if k >= j:
                    elem = [0]*(self.polynomial_degree+1)
                    elem_err = [0]*(self.polynomial_degree+1)
                else:
                    omega = lnxj - lnxk
                    for poly_power in range(self.polynomial_degree +1):
                        ker = self.get_ns_ker(constants, nf, delta_t,j,k,poly_power)
                        i,err = integrate.quad(ker,0,np.inf,weight='cos',wvar=omega,epsabs=1e-25,epsrel=1e-6,limlst=100,limit=100)
                        print(f"{j: 2d} {k: 2d} {poly_power} with {lnxj:.3e},{lnxk:.3e}->{omega:.3e} => {i:+.5e}+-{err:.3e}")
                        elem.append(i)
                        elem_err.append(err)
                line.append(elem)
                line_err.append(elem_err)
            raw.append(line)
            raw_errors.append(line_err)
        return np.array(raw), np.array(raw_errors)

    def get_ns(self, constants, nf, delta_t):
        """
            Computes non-singlet operator.
        """
        raw, raw_err = self.get_raw_ns(constants,nf,delta_t)
        op = []
        op_err = []
        for j in range(len(self.xgrid)):
            line = []
            line_err = []
            bf = self.basis[j]
            for k in range(len(self.xgrid)):
                el = 0
                el_err = 0
                for A in bf.areas:
                    Amax = A.lower_index + 1
                    Amin = A.lower_index
                    cs = A.coefs
                    polys = raw[Amax][k] - raw[Amin][k]
                    polys_err = raw_err[Amax][k] - raw_err[Amin][k]
                    el += np.matmul(cs,polys)
                    el_err += np.matmul(cs,polys_err)
                line.append(el)
                line_err.append(np.abs(el_err))
            op.append(line)
            op_err.append(line_err)
        return np.array(op).T, np.array(op_err).T
