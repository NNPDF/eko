r"""The polarized, time-like |OME|."""

import numba as nb


@nb.njit(cache=True)
def A_non_singlet(_matching_order, _n, _nf, _L):
    raise NotImplementedError("Polarised, time-like is not yet implemented")


@nb.njit(cache=True)
def A_singlet(_matching_order, _n, _nf, _L, _is_msbar):
    raise NotImplementedError("Polarised, time-like is not yet implemented")
