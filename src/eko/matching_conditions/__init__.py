# -*- coding: utf-8 -*-
"""This module defines the matching conditions for the |VFNS| evolution."""

import numba as nb
import numpy as np

from .. import basis_rotation as br
from .. import member
from . import as1, as2, as3


@nb.njit(cache=True)
def A_singlet_as3(n, sx, L, is_msbar, nf, sx_ns):
    r"""Computes the tower of the |N3LO| singlet |OME|.

    Parameters
    ----------
    n : complex
        Mellin variable
    sx : list
        singlet like harmonic sums cache
    L : float
        :math:`\log(q^2/m_h^2)`
    is_msbar: bool
        add the |MSbar| contribution
    nf : int
        number of active flavor below threshold
    sx_ns : list
        non-singlet like harmonic sums cache

    Returns
    -------
    numpy.ndarray
        |N3LO| singlet |OME|

    See Also
    --------
        eko.matching_conditions.as1.A_singlet : :math:`A^{S,(1)}(N)`
        eko.matching_conditions.as2.A_singlet : :math:`A_{S,(2)}(N)`
        eko.matching_conditions.a32.A_singlet : :math:`A_{S,(3)}(N)`
    """
    A_s = np.zeros((3, 3, 3), np.complex_)
    A_s[0] = as1.A_singlet(n, sx, L)
    A_s[1] = as2.A_singlet(n, sx, L, is_msbar)
    A_s[2] = as3.A_singlet(n, sx, sx_ns, nf, L)
    return A_s


@nb.njit(cache=True)
def A_singlet_as2(n, sx, L, is_msbar, _nf=0, _sx_ns=None):
    r"""Computes the tower of the |NNLO| singlet |OME|.

    Parameters
    ----------
    n : complex
        Mellin variable
    sx : list
        singlet like harmonic sums cache
    L : float
        :math:`\log(q^2/m_h^2)`
    is_msbar: bool
        add the |MSbar| contribution
    nf : int
        number of active flavor below threshold
    sx_ns : list
        non-singlet like harmonic sums cache

    Returns
    -------
    numpy.ndarray
        |NNLO| singlet |OME|

    See Also
    --------
        eko.matching_conditions.as1.A_singlet : :math:`A^{S,(1)}(N)`
        eko.matching_conditions.as2.A_singlet : :math:`A_{S,(2)}(N)`
    """
    A_s = np.zeros((2, 3, 3), np.complex_)
    A_s[0] = as1.A_singlet(n, sx, L)
    A_s[1] = as2.A_singlet(n, sx, L, is_msbar)
    return A_s


@nb.njit(cache=True)
def A_singlet_as1(n, sx, L, _is_msbar=False, _nf=0, _sx_ns=None):
    r"""Computes the tower of the |NLO| singlet |OME|.

    Parameters
    ----------
    n : complex
        Mellin variable
    sx : list
        singlet like harmonic sums cache
    L : float
        :math:`\log(q^2/m_h^2)`
    is_msbar: bool
        add the |MSbar| contribution
    nf : int
        number of active flavor below threshold
    sx_ns : list
        non-singlet like harmonic sums cache

    Returns
    -------
    numpy.ndarray
        |NLO| singlet |OME|

    See Also
    --------
        eko.matching_conditions.as1.A_singlet : :math:`A^{S,(1)}(N)`
    """
    A_s = np.zeros((1, 3, 3), np.complex_)
    A_s[0] = as1.A_singlet(n, sx, L)
    return A_s


@nb.njit(cache=True)
def A_non_singlet_as3(n, sx, L, nf):
    r"""Computes the tower of the |N3LO| non-singlet |OME|.

    Parameters
    ----------
    n : complex
        Mellin variable
    sx : list
        harmonic sums cache
    L : float
        :math:`\log(q^2/m_h^2)`
    nf : int
        number of active flavor below threshold

    Returns
    -------
    numpy.ndarray
            |N3LO| non-singlet |OME|

    See Also
    --------
        eko.matching_conditions.as1.A_ns : :math:`A_{ns}^{(1)}(N)`
        eko.matching_conditions.as2.A_ns : :math:`A_{ns}^{(2)}(N)`
        eko.matching_conditions.as3.A_ns : :math:`A_{ns}^{(3)}(N)`
    """
    A_ns = np.zeros((3, 2, 2), np.complex_)
    A_ns[0] = as1.A_ns(n, sx, L)
    A_ns[1] = as2.A_ns(n, sx, L)
    A_ns[2] = as3.A_ns(n, sx, nf, L)
    return A_ns


@nb.njit(cache=True)
def A_non_singlet_as2(n, sx, L, _nf=0):
    r"""Computes the tower of the |NNLO| non-singlet |OME|.

    Parameters
    ----------
    n : complex
        Mellin variable
    sx : list
        harmonic sums cache
    L : float
        :math:`\log(q^2/m_h^2)`
    nf : int
        number of active flavor below threshold

    Returns
    -------
    numpy.ndarray
            |NNLO| non-singlet |OME|

    See Also
    --------
        eko.matching_conditions.as1.A_ns : :math:`A_{ns}^{(1)}(N)`
        eko.matching_conditions.as2.A_ns : :math:`A_{ns}^{(2)}(N)`
    """
    A_ns = np.zeros((2, 2, 2), np.complex_)
    A_ns[0] = as1.A_ns(n, sx, L)
    A_ns[1] = as2.A_ns(n, sx, L)
    return A_ns


@nb.njit(cache=True)
def A_non_singlet_as2(n, sx, L, _nf=0):
    r"""Computes the tower of the |NLO| non-singlet |OME|.

    Parameters
    ----------
    n : complex
        Mellin variable
    sx : list
        harmonic sums cache
    L : float
        :math:`\log(q^2/m_h^2)`
    nf : int
        number of active flavor below threshold

    Returns
    -------
    numpy.ndarray
            |NLO| non-singlet |OME|

    See Also
    --------
        eko.matching_conditions.as1.A_ns : :math:`A_{ns}^{(1)}(N)`
    """
    A_ns = np.zeros((1, 2, 2), np.complex_)
    A_ns[0] = as1.A_ns(n, sx, L)
    return A_ns


class MatchingCondition(member.OperatorBase):
    """
    Matching conditions for |PDF| at threshold.

    The continuation of the (formally) heavy non-singlet distributions
    with either the full singlet :math:`S` or the full valence :math:`V`
    is considered "trivial". Instead at |NNLO| additional terms ("non-trivial")
    enter.
    """

    @classmethod
    def split_ad_to_evol_map(
        cls,
        op_members,
        nf,
        q2_thr,
        intrinsic_range,
    ):
        """
        Create the instance from the |OME|.

        Parameters
        ----------
            op_members : eko.operator_matrix_element.OperatorMatrixElement.op_members
                Attribute of :class:`~eko.operator_matrix_element.OperatorMatrixElement`
                containing the |OME|
            nf : int
                number of active flavors *below* the threshold
            q2_thr: float
                threshold value
            intrinsic_range : list
                list of intrinsic quark pids
        """

        m = {
            "S.S": op_members[(100, 100)],
            "S.g": op_members[(100, 21)],
            "g.S": op_members[(21, 100)],
            "g.g": op_members[(21, 21)],
            "V.V": op_members[(200, 200)],
        }

        # add elements which are already active
        for f in range(2, nf + 1):
            n = f**2 - 1
            m[f"V{n}.V{n}"] = m["V.V"]
            m[f"T{n}.T{n}"] = m["V.V"]

        # activate the next heavy quark
        hq = br.quark_names[nf]
        m.update(
            {
                # f"{hq}-.V": op_members[(br.matching_hminus_pid, 200)],
                f"{hq}+.S": op_members[(br.matching_hplus_pid, 100)],
                f"{hq}+.g": op_members[(br.matching_hplus_pid, 21)],
            }
        )

        # intrinsic matching
        if len(intrinsic_range) != 0:
            op_id = member.OpMember.id_like(op_members[(200, 200)])
            for intr_fl in intrinsic_range:
                ihq = br.quark_names[intr_fl - 1]  # find name
                if intr_fl > nf + 1:
                    # keep the higher quarks as they are
                    m[f"{ihq}+.{ihq}+"] = op_id.copy()
                    m[f"{ihq}-.{ihq}-"] = op_id.copy()
                elif intr_fl == nf + 1:
                    # match the missing contribution from h+ and h-
                    m.update(
                        {
                            f"{ihq}+.{ihq}+": op_members[
                                (br.matching_hplus_pid, br.matching_hplus_pid)
                            ],
                            f"S.{ihq}+": op_members[(100, br.matching_hplus_pid)],
                            f"g.{ihq}+": op_members[(21, br.matching_hplus_pid)],
                            f"{ihq}-.{ihq}-": op_members[
                                (br.matching_hminus_pid, br.matching_hminus_pid)
                            ],
                            # f"V.{ihq}-": op_members[(200, br.matching_hminus_pid)],
                        }
                    )
        return cls.promote_names(m, q2_thr)
