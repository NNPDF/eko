r"""The polarized, space-like |OME|."""

import numba as nb


@nb.njit(cache=True)
def A_non_singlet(_matching_order, _n, _sx, _nf, _L):
    raise NotImplementedError("Polarised, space-like is not yet implemented")


@nb.njit(cache=True)
def A_singlet(_matching_order, _n, _sx, _nf, _L, _is_msbar, _sx_ns=None):
    raise NotImplementedError("Polarised, space-like is not yet implemented")
