# -*- coding: utf-8 -*-
"""
This file provides all necessary tools for PDF interpolation.

The files provides the tools for generating grids and the
Lagrange interpolation polynomials

"""
import math
import numpy as np
import numba as nb
from eko import t_float

#nb.njit = lambda x: x

#TODO deprecated
def get_Lagrange_basis_functions(xgrid_in, polynom_rank: int):
    """Setup all basis function for the interpolation

    Parameters
    ----------
      xgrid_in : array
        Grid in x-space from which the interpolaters are constructed
      polynom_rank : int
        degree of the interpolation polynomial

    Returns
    -------
      list_of_basis_functions : array
        list with configurations for all basis functions
    """
    # setup params
    xgrid = np.unique(xgrid_in)
    xgrid_size = len(xgrid)
    if not len(xgrid_in) == xgrid_size:
        raise ValueError("xgrid is not unique")
    if xgrid_size < 2:
        raise ValueError("xgrid needs at least 2 points")
    if polynom_rank < 1:
        raise ValueError("need at least linear interpolation")
    if xgrid_size < polynom_rank:
        raise ValueError(
            f"to interpolate with rank {polynom_rank} we need at"
            + "least that much points"
        )

    # create blocks
    list_of_blocks = []
    for j in range(xgrid_size - 1):
        kmin = max(0, j - polynom_rank // 2)  # borders are (]
        kmax = kmin + polynom_rank
        if kmax >= xgrid_size:
            kmax = xgrid_size - 1
            kmin = kmax - polynom_rank
        list_of_blocks.append((kmin, kmax))

    # setup basis functions
    list_of_basis_functions = []
    for j in range(xgrid_size):
        areas = []
        for k in range(xgrid_size - 1):
            areas.append({"lower_index": k, "reference_indices": None})
        list_of_basis_functions.append({"polynom_number": j, "areas": areas})

    for j, current_block in enumerate(list_of_blocks):
        for k in range(current_block[0], current_block[1] + 1):
            list_of_basis_functions[k]["areas"][j]["reference_indices"] = current_block

    # compute coefficients
    def is_not_zero_sector(e):
        return not e["reference_indices"] is None

    for j, current_polynom in enumerate(list_of_basis_functions):
        # clean up zero sectors
        current_polynom["areas"] = list(
                filter(is_not_zero_sector, current_polynom["areas"])
                )
        # precompute coefficients
        xj = xgrid[j]
        for current_area in current_polynom["areas"]:
            denominator = 1.0
            coeffs = np.array([1])
            for k in range(
                current_area["reference_indices"][0],
                current_area["reference_indices"][1] + 1,
            ):
                if k == j:
                    continue
                xk = xgrid[k]
                # Lagrange interpolation formula
                denominator *= xj - xk
                x_coeffs = np.insert(coeffs, 0, 0)
                Mxk_coeffs = -xk * coeffs
                Mxk_coeffs = np.append(
                    Mxk_coeffs, np.zeros(len(x_coeffs) - len(Mxk_coeffs))
                )
                coeffs = x_coeffs + Mxk_coeffs
            # apply common denominator
            coeffs = coeffs / denominator
            # save in dictionary
            current_area["coeffs"] = coeffs
            current_area["xmin"] = xgrid[current_area["lower_index"]]
            current_area["xmax"] = xgrid[current_area["lower_index"] + 1]
            # clean up
            # we still need polynom_number as the first polynom has borders [], to allow for testing
    # return all functions
    return list_of_basis_functions

#TODO: deprecated
def evaluate_Lagrange_basis_function_log_N(N, conf, lnx):
    """Get a single, logarithmic Lagrange interpolator in N-space multiplied
    by the Mellin-inversion factor.

    .. math::
      \\tilde P^{\\ln}(N)*exp(- N * log(x))

    The polynomials contain naturally factors of :math:`exp(N * log(x_{min/max}))`
    which can be joined with the Mellin inversion factor.

    Parameters
    ----------
      N : t_float
        Evaluated point in N-space
      conf : dict
        dictionary of values for the coefficients of the interpolator
      lnx : t_float
        Mellin-inversion point :math:`log(x)`

    Returns
    -------
      p(N)*x^{-N} : t_complex
        Evaluated polynom at N times x^{-N}
    """
    if not "areas" in conf or len(conf["areas"]) <= 0:
        raise ValueError("need some areas to explore")

    # helper functions
    def get1(j, u):
        s = 0.0
        for k in range(0, j + 1):
            s += (-u) ** (k) * np.math.factorial(j) / np.math.factorial(k)
        return (-1) ** j * s

    def get2(j, lnxminmax):
        return np.exp(N * (lnxminmax - lnx)) * get1(j, N * lnxminmax)

    # sum all areas
    res = 0.0
    for current_area in conf["areas"]:
        for j, coeff in enumerate(current_area["coeffs"]):
            logxmax = current_area["xmax"]
            logxmin = current_area["xmin"]
            c = get2(j, logxmax) - get2(j, logxmin)
            res += coeff * c / N ** (1 + j)
    return res


def get_xgrid_linear_at_id(grid_size: int, xmin: t_float = 0.0, xmax: t_float = 1.0):
    """Computes a linear grid on true x - corresponds to the flag `linear@id`

    This function is mainly for testing purpuse, as it is not physically relevant.

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
    # numba does not accept to set the type of the grid
    xgrid = np.linspace(xmin, xmax, grid_size)
    return xgrid


def get_xgrid_linear_at_log(grid_size: int, xmin: t_float, xmax: t_float = 1.0):
    """Computes a linear grid on log(x) - corresponds to the flag `linear@log`

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
    return np.logspace(np.log10(xmin), np.log10(xmax), num=grid_size, dtype=t_float)

# Compiled functions
@nb.njit
def helper_evaluate_N(i, x, xref, N):
    exp_arg = N*(xref - x)
    exp_val = np.exp(exp_arg)
    facti = math.gamma(i+1)
    # Aux values
    u = -N*xref
    res_sum = 0.0
    for k in range(i+1):
        res_sum += pow(u,k) / math.gamma(k+1)*facti
    res_sum *= pow(-1, i)
    return exp_val * res_sum

# Interpolator Classes
class Area:

    def __init__(self, lower_index, polynom_number, block, xgrid):
        self.poly_number = polynom_number
        self._reference_indices = block
        self.kmin = block[0]
        self.kmax = block[1]
        self.coefs = self.compute_coefs(xgrid)
        self.xmin = xgrid[lower_index]
        self.xmax = xgrid[lower_index+1]

    @property
    def reference_indices(self):
        return self._reference_indices

    @reference_indices.getter
    def reference_indices(self):
        for k in range(self.kmin, self.kmax+1):
            if k != self.poly_number:
                yield k

    def compute_coefs(self, xgrid):
        denominator = 1.0
        coeffs = np.ones(1)
        xj = xgrid[self.poly_number]
        for s, k in enumerate(self.reference_indices):
            xk = xgrid[k]
            denominator *= xj - xk
            Mxk_coeffs = -xk * coeffs
            coeffs = np.concatenate(([0.0], coeffs))
            coeffs[:s++1] += Mxk_coeffs
        coeffs /= denominator
        return coeffs

    def __iter__(self):
        for coef in self.coefs:
            yield coef


class BasisFunction:
    """
        Object containing
        a list of areas for a given polynomial number
        defined by (xmin-xmax) and containing a list
        of coefficients.
    """

    def __init__(self, poly_number, list_of_blocks, xgrid):
        self.poly_number = poly_number
        self.areas = []

        for i, block in enumerate(list_of_blocks):
            if block[0] <= poly_number <= block[1]:
                new_area = Area(i, self.poly_number, block, xgrid)
                self.areas.append(new_area)

        self.callable = None
        self.compile()


    def get_limits(self):
        limits = []
        for area in self:
            limits.append( (area.xmin, area.xmax) )
        return limits

    def get_coefs(self):
        coefs = []
        for area in self:
            coefs.append( area.coefs )
        return coefs

    def areas_to_const(self):
        area_list = []
        for area in self:
            area_list.append( (area.xmin, area.xmax, area.coefs) )
        return tuple(area_list)

    def compile(self):
        area_list = self.areas_to_const()
        def evaluate_Nx(N, x):
            res = 0.0
            for xmin, xmax, coefs in area_list:
                for i, coef in enumerate(coefs):
                    c = helper_evaluate_N(i, x, xmax, N) - helper_evaluate_N(i, x, xmin, N)
                    res += coef*c / pow(N, 1+i)
            return res
        self.callable = nb.njit(evaluate_Nx)

    def __iter__(self):
        for area in self.areas:
            yield area

    def __call__(self, N, x):
        return self.callable(N, x)


class InterpolatorDispatcher:

    def __init__(self, xgrid, polynom_rank, log = False):

        # TODO: copy the checks here

        if log:
            xgrid = np.log(xgrid)

        # Save the different variables
        self.xgrid = xgrid
        self.polynom_rank = polynom_rank
        self.log = log

        # Define some useful libraries
        xgrid_size = len(xgrid)


        # Create blocks
        list_of_blocks = []
        po2 = polynom_rank // 2
        for i in range(xgrid_size-1):
            kmin = max(0, i - po2)
            kmax = kmin + polynom_rank
            if kmax >= xgrid_size:
                kmax = xgrid_size - 1
                kmin = kmax - polynom_rank
            list_of_blocks.append((kmin, kmax))

        # Generate the basis functions
        basis_functions = []
        for i in range(xgrid_size):
            new_basis = BasisFunction(i, list_of_blocks, xgrid)
            basis_functions.append(new_basis)
        self.basis = basis_functions

    def __iter__(self):
        for basis in self.basis:
            yield basis
























# Not used at the moment
def evaluate_Lagrange_basis_function_x(x, conf):
    """Get a single Lagrange interpolator in x-space

    .. math::
      \\tilde P(x)

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
    if not "areas" in conf or len(conf["areas"]) <= 0:
        raise ValueError("need some areas to explore")
    # search
    for current_area in conf["areas"]:
        # borders are usually (] - except for the first
        if conf["polynom_number"] == 0:
            if x < current_area["xmin"]:
                continue
        else:
            if x <= current_area["xmin"]:
                continue
        if x > current_area["xmax"]:
            continue
        # match found
        res = 0.0
        for k, coeff in enumerate(current_area["coeffs"]):
            res += coeff * x ** k
        return res
    # no match
    return 0.0

def evaluate_Lagrange_basis_function_N(N, conf, lnx):
    """Get a single Lagrange interpolator in N-space multiplied
    by the Mellin-inversion factor.

    .. math::
      \\tilde P^{\\ln}(N)*exp(- N * log(x))

    The polynomials contain naturally factors of :math:`exp(N * j * log(x_{min/max}))`
    which can be joined with the Mellin inversion factor.

    Parameters
    ----------
      N : t_float
        Evaluated point in N-space
      conf : dict
        dictionary of values for the coefficients of the interpolator
      lnx : t_float
        Mellin-inversion point :math:`log(x)`

    Returns
    -------
      p(N)*x^{-N} : t_complex
        Evaluated polynom at N times x^{-N}
    """
    if not "areas" in conf or len(conf["areas"]) <= 0:
        raise ValueError("need some areas to explore")
    res = 0.0
    for current_area in conf["areas"]:
        for j, coeff in enumerate(current_area["coeffs"]):
            if current_area["xmin"] == 0.0:
                low = 0.0
            else:
                lnxmin = np.log(current_area["xmin"])
                low = np.exp(N * (lnxmin - lnx) + j * lnxmin)
            lnxmax = np.log(current_area["xmax"])
            up = np.exp(N * (lnxmax - lnx) + j * lnxmax)
            res += coeff * (up - low) / (N + j)
    return res

def get_Lagrange_basis_functions_log(xgrid_in, polynom_rank: int):
    """Setup all basis function for logarithmic interpolation

    See Also
    --------
      get_Lagrange_basis_functions
    """
    return get_Lagrange_basis_functions(np.log(xgrid_in), polynom_rank)


def evaluate_Lagrange_basis_function_log_x(x, conf):
    """Get a single, logarithmic Lagrange interpolator in x-space

    .. math::
      \\tilde P^{\\ln}(x)

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
    return evaluate_Lagrange_basis_function_x(np.log(x), conf)

if __name__ == "__main__":
    xgrid = np.logspace(-3, 0, 10)
    polrank = 4

    reference = get_Lagrange_basis_functions(xgrid, polrank)
    mine = InterpolatorDispatcher(xgrid, polrank, False)

    # Check that the basis is the same
    for ref, new in zip(reference, mine.basis):
        assert ref['polynom_number'] == new.poly_number
        ref_areas = ref['areas']
        for ref_area, new_area in zip(ref_areas, new):
            assert ref_area['xmin'] == new_area.xmin
            assert ref_area['xmax'] == new_area.xmax
            for ref_coef, new_coef in zip(ref_area['coeffs'], new_area):
                assert ref_coef == new_coef
    # Check that the results are the same
    for i,j in np.random.rand(10, 2):
        N = complex(i,0.0)
        for lnx in np.random.rand(10):
            for ref_poly, new_poly in zip(reference, mine):
                ref_res = evaluate_Lagrange_basis_function_log_N(N, ref_poly, lnx)
                new_res = new_poly(N, lnx)
                np.testing.assert_allclose(np.real(ref_res), np.real(new_res), rtol=1e-4)
