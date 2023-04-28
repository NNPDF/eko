"""Module containing the harmonics sums implementation.

Definitions are coming from :cite:`MuselliPhD,Bl_mlein_2000,Blumlein:2009ta`.
"""
from . import cache, g_functions, polygamma
from .w1 import S1, Sm1
from .w2 import S2, Sm2
from .w3 import S3, S21, S2m1, Sm2m1, Sm3, Sm21
from .w4 import S4, S31, S211, Sm4, Sm22, Sm31, Sm211
from .w5 import S5, Sm5

# TODO: CLEAN
# @nb.njit(cache=True)
# def base_harmonics_cache(n, is_singlet, max_weight=5, n_max_sums_weight=7):
#     r"""Get the harmonics sums S basic cache.

#     Only single index harmonics are computed and stored
#     in the first (:math:`S_{n}`) or in the last column (:math:`S_{-n}`).

#     Multi indices harmonics sums can be stored in the middle columns.

#     Parameters
#     ----------
#     n : complex
#         Mellin moment
#     is_singlet : bool
#         symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N = 1`),
#         False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)
#     max_weight : int
#         max harmonics weight, max value 5 (default)
#     n_max_sums_weight : int
#         max number of harmonics sums for a given weight

#     Returns
#     -------
#     np.ndarray
#         list of harmonics sums: (weights, n_max_sums_weight)

#     """
#     h_cache = np.zeros((max_weight, n_max_sums_weight), dtype=np.complex_)
#     h_cache[:, 0] = sx(n, max_weight)
#     if n_max_sums_weight > 1:
#         h_cache[:, -1] = smx(n, h_cache[:, 0], is_singlet)
#     return h_cache


# @nb.njit(cache=True)
# def compute_cache(n, max_weight, is_singlet):
#     r"""Get the harmonics sums cache.

#     Parameters
#     ----------
#     n : complex
#         Mellin moment
#     max_weight : int
#         maximum weight to compute [2,3,4,5]
#     is_singlet : bool
#         symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N = 1`),
#         False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

#     Returns
#     -------
#     list
#         harmonic sums cache. At |N3LO| it contains:

#         .. math ::
#             [[S_1,S_{-1}],
#             [S_2,S_{-2}],
#             [S_{3}, S_{2,1}, S_{2,-1}, S_{-2,1}, S_{-2,-1}, S_{-3}],
#             [S_{4}, S_{3,1}, S_{2,1,1}, S_{-2,-2}, S_{-3, 1}, S_{-4}],]
#             [S_{5}, S_{-5}]

#     """
#     # max number of harmonics sum of a given weight for a given max weight.
#     n_max_sums_weight = {2: 1, 3: 3, 4: 7, 5: 7}
#     sx = base_harmonics_cache(n, is_singlet, max_weight, n_max_sums_weight[max_weight])
#     if max_weight == 3:
#         # Add Sm21 to cache
#         sx[2, 1] = Sm21(n, sx[0, 0], sx[0, -1], is_singlet)
#     elif max_weight >= 4:
#         # Add weight 3 and 4 to cache
#         sx[2, 1:-2] = s3x(n, sx[:, 0], sx[:, -1], is_singlet)
#         sx[3, 1:-1] = s4x(n, sx[:, 0], sx[:, -1], is_singlet)
#     # return list of list keeping the non zero values
#     return [[el for el in sx_list if el != 0] for sx_list in sx]


# @nb.njit(cache=True)
# def compute_qed_ns_cache(n, s1):
#     r"""Get the harmonics sums cache needed for the qed non singlet AD.

#     Parameters
#     ----------
#     n : complex
#         Mellin moment
#     s1 : float
#         harmonic sum :math:`S_1(N)`

#     Returns
#     -------
#     list
#         harmonic sums cache. It contains:

#         .. math ::
#             [S_1(n/2),
#             S_2(n/2),
#             S_{3}(n/2),
#             S_1((n+1)/2),
#             S_2((n+1)/2),
#             S_{3}((n+1)/2),
#             g_3(n),
#             g_3(n+2)]

#     """
#     s1h = S1(n / 2.0)
#     sx_qed_ns = [s1h]
#     S2h = S2(n / 2.0)
#     sx_qed_ns.append(S2h)
#     S3h = S3(n / 2.0)
#     sx_qed_ns.append(S3h)
#     S1p1h = S1((n + 1.0) / 2.0)
#     sx_qed_ns.append(S1p1h)
#     S2p1h = S2((n + 1.0) / 2.0)
#     sx_qed_ns.append(S2p1h)
#     S3p1h = S3((n + 1.0) / 2.0)
#     sx_qed_ns.append(S3p1h)
#     g3N = g_functions.mellin_g3(n, s1)
#     sx_qed_ns.append(g3N)
#     S1p2 = polygamma.recursive_harmonic_sum(s1, n, 2, 1)
#     g3Np2 = g_functions.mellin_g3(n + 2.0, S1p2)
#     sx_qed_ns.append(g3Np2)
#     return sx_qed_ns
