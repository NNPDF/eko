# -*- coding: utf-8 -*-
"""
This module contains some harmonics sum.
Definitions are coming from :cite:`MuselliPhD,Bl_mlein_2000,Blumlein:2009ta`
"""
import numba as nb
import numpy as np

from .w1 import S1, Sm1
from .w2 import S2, Sm2
from .w3 import S3, S21, S2m1, Sm2m1, Sm3, Sm21
from .w4 import S4, S31, S211, Sm4, Sm22, Sm31, Sm211
from .w5 import S5, S23, S41, S221, S311, S2111, S2m3, S21m2, Sm5, Sm23, Sm221, Sm2111


@nb.njit(cache=True)
def base_harmonics_cache(n, is_singlet, max_weight=5, n_max_sums_weight=7):
    r"""
    Get the harmonics sums S basic cache.
    Only single index harmonics are computed and stored
    in the first (:math:`S_{n}`) or in the last column (:math:`S_{-n}`)

    Multi indices harmonics sums can be stored in the middle columns.

    Parameters
    ----------
        n : complex
            Mellin moment
        is_singlet: bool
            symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N = 1`),
            False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)
        max_weight : int
            max harmonics weight, max value 5 (default)
        n_max_sums_weight : int
            max number of harmonics sums for a given weight

    Returns
    -------
        h_cache : np.ndarray
            list of harmonics sums: (weights, n_max_sums_weight)
    """
    h_cache = np.zeros((max_weight, n_max_sums_weight), dtype=np.complex_)
    h_cache[:, 0] = sx(n, max_weight)
    if n_max_sums_weight > 1:
        h_cache[:, -1] = smx(n, h_cache[:, 0], is_singlet)
    return h_cache


@nb.njit(cache=True)
def sx(n, max_weight=5):
    """
    Get the harmonics sums S cache

    Parameters
    ----------
        n : complex
            Mellin moment
        max_weight : int
            max harmonics weight, max value 5 (default)

    Returns
    -------
        sx : np.ndarray
            list of harmonics sums (:math:`S_{1,..,w}`)
    """
    sx = np.zeros(max_weight, dtype=np.complex_)
    if max_weight >= 1:
        sx[0] = S1(n)
    if max_weight >= 2:
        sx[1] = S2(n)
    if max_weight >= 3:
        sx[2] = S3(n)
    if max_weight >= 4:
        sx[3] = S4(n)
    if max_weight >= 5:
        sx[4] = S5(n)
    return sx


@nb.njit(cache=True)
def smx(n, sx, is_singlet):
    r"""
    Get the harmonics S-minus cache

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : numpy.ndarray
            List of harmonics sums: :math:`S_{1},\dots,S_{w}`
        is_singlet: bool
            symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N = 1`),
            False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)
    Returns
    -------
        smx : np.ndarray
            list of harmonics sums (:math:`S_{-1,..,-w}`)
    """
    max_weight = sx.size
    smx = np.zeros(max_weight, dtype=np.complex_)
    if max_weight >= 1:
        smx[0] = Sm1(n, sx[0], is_singlet)
    if max_weight >= 2:
        smx[1] = Sm2(n, sx[1], is_singlet)
    if max_weight >= 3:
        smx[2] = Sm3(n, sx[2], is_singlet)
    if max_weight >= 4:
        smx[3] = Sm4(n, sx[3], is_singlet)
    if max_weight >= 5:
        smx[4] = Sm5(n, sx[4], is_singlet)
    return smx


@nb.njit(cache=True)
def s3x(n, sx, smx, is_singlet):
    r"""
    Compute the weight 3 multi indices harmonics sums cache

    Parameters
    ----------
        n: complex
            Mellin moment
        sx : list
            List of harmonics sums: :math:`S_{1},S_{2}`
        smx : list
            List of harmonics sums: :math:`S_{-1},S_{-2}`
        is_singlet: bool
            symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N = 1`),
            False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
        s3x: np.ndarray
            list containing: :math:`S_{2,1},S_{2,-1},S_{-2,1},S_{-2,-1}`
    """
    return np.array(
        [
            S21(n, sx[0], sx[1]),
            S2m1(n, sx[1], smx[0], smx[1], is_singlet),
            Sm21(n, sx[0], smx[0], is_singlet),
            Sm2m1(n, sx[0], sx[1], smx[1]),
        ]
    )


@nb.njit(cache=True)
def s4x(n, sx, smx, is_singlet):
    r"""
    Compute the weight 4 multi indices harmonics sums cache

    Parameters
    ----------
        n: complex
            Mellin moment
        sx : list
            List of harmonics sums: :math:`S_{1},S_{2},S_{3},S_{4}`
        smx : list
            List of harmonics sums: :math:`S_{-1},S_{-2}`
        is_singlet: bool
            symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N = 1`),
            False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
        s4x: np.ndarray
            list containing: :math:`S_{3,1},S_{2,1,1},S_{-2,2},S_{-2,1,1},S_{-3,1}`
    """
    sm31 = Sm31(n, sx[0], smx[0], smx[1], is_singlet)
    return np.array(
        [
            S31(n, sx[0], sx[1], sx[2], sx[3]),
            S211(n, sx[0], sx[1], sx[2]),
            Sm22(n, sx[0], sx[1], smx[1], sm31, is_singlet),
            Sm211(n, sx[0], sx[1], smx[0], is_singlet),
            sm31,
        ]
    )
