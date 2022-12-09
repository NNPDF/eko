r"""Holds the classes that define the |FNS|."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


class PathSegment:
    """Oriented path in the threshold landscape.

    Attributes
    ----------
    q2_from : float
        starting point
    q2_to : float
        final point
    nf : int
        number of active flavors

    Parameters
    ----------
    q2_from : float
        starting point
    q2_to : float
        final point
    nf : int
        number of active flavors
    """

    def __init__(self, q2_from, q2_to, nf):
        self.q2_from = q2_from
        self.q2_to = q2_to
        self.nf = nf

    @property
    def is_downward_q2(self):
        """Return True if ``q2_from`` is bigger than ``q2_to``."""
        return self.q2_from > self.q2_to

    @property
    def tuple(self):
        """Return tuple representation suitable for hashing."""
        return (self.q2_from, self.q2_to, self.nf)

    def __repr__(self):
        """Return string representation."""
        return f"PathSegment({self.q2_from} -> {self.q2_to}, nf={self.nf})"


class ThresholdsAtlas:
    r"""Holds information about the matching scales.

    These scales are the :math:`Q^2` has to pass in order to get there from a
    given :math:`Q^2_{ref}`.

    """

    def __init__(
        self,
        masses,
        q2_ref=None,
        nf_ref=None,
        thresholds_ratios=None,
        max_nf=None,
    ):
        """Create basic atlas.

        Parameters
        ----------
        masses : list(float)
            list of quark masses squared
        q2_ref : float
            reference scale
        nf_ref : int
            number of active flavors at the reference scale
        thresholds_ratios : list(float)
            list of ratios between matching scales and masses squared
        max_nf : int
            maximum number of active flavors
        """
        masses = list(masses)
        if masses != sorted(masses):
            raise ValueError("masses need to be sorted")
        # combine them
        thresholds = self.build_area_walls(masses, thresholds_ratios, max_nf)
        self.area_walls = [0] + thresholds + [np.inf]

        # check nf_ref
        if nf_ref is not None:
            if q2_ref is None:
                raise ValueError(
                    "Without a reference Q2 value a reference number of flavors "
                    "does not make sense!"
                )
            # else self.q2_ref is not None
            nf_init = 2 + len(list(filter(lambda x: np.isclose(0, x), self.area_walls)))
            if nf_ref < nf_init:
                raise ValueError(
                    f"The reference number of flavors is set to {nf_ref}, "
                    f"but the atlas starts at {nf_init}"
                )
            nf_final = 2 + len(list(filter(lambda x: x < np.inf, self.area_walls)))
            if nf_ref > nf_final:
                raise ValueError(
                    f"The reference number of flavors is set to {nf_ref}, "
                    f"but the atlas stops at {nf_final}"
                )

        # Init values
        self.q2_ref = q2_ref
        self.nf_ref = nf_ref
        self.thresholds_ratios = thresholds_ratios
        logger.info(str(self))

    def __repr__(self):
        """Return string representation."""
        walls = " - ".join([f"{w:.2e}" for w in self.area_walls])
        return f"ThresholdsAtlas [{walls}], ref={self.q2_ref} @ {self.nf_ref}"

    @classmethod
    def ffns(cls, nf, q2_ref=None):
        """Create a |FFNS| setup.

        The function creates simply sufficient thresholds at ``0`` (in the
        beginning), since the number of flavors is determined by counting
        from below.

        Parameters
        ----------
        nf : int
            number of light flavors
        q2_ref : float
            reference scale
        """
        return cls([0] * (nf - 3) + [np.inf] * (6 - nf), q2_ref)

    @staticmethod
    def build_area_walls(masses, thresholds_ratios=None, max_nf=None):
        r"""Create the object from the informations on the run card.

        The thresholds are computed by :math:`(m_q \cdot k_q^{Thr})`.

        Parameters
        ----------
        masses : list(float)
            heavy quark masses squared
        thresholds_ratios : list(float)
            list of ratios between matching scales and masses squared
        max_nf : int
            maximum number of flavors

        Returns
        -------
        list
            threshold list
        """
        if len(masses) != 3:
            raise ValueError("There have to be 3 quark masses")
        if thresholds_ratios is None:
            thresholds_ratios = (1.0, 1.0, 1.0)
        if len(thresholds_ratios) != 3:
            raise ValueError("There have to be 3 quark threshold ratios")
        if max_nf is None:
            max_nf = 6

        thresholds = []
        for m, k in zip(masses, thresholds_ratios):
            thresholds.append(m * k)
        # cut array = simply reduce some thresholds
        thresholds = thresholds[: max_nf - 3]
        return thresholds

    def path(self, q2_to, nf_to=None, q2_from=None, nf_from=None):
        """Get path from ``q2_from`` to ``q2_to``.

        Parameters
        ----------
        q2_to: float
            target value of q2
        q2_from: float
            starting value of q2

        Returns
        -------
        list(PathSegment)
            List of :class:`PathSegment` to go through in order to get from ``q2_from``
            to ``q2_to``.
        """
        # fallback to init config
        if q2_from is None:
            q2_from = self.q2_ref
        if nf_from is None:
            nf_from = self.nf_ref
        # determine reference thresholds
        if nf_from is None:
            nf_from = 2 + np.digitize(q2_from, self.area_walls)
        if nf_to is None:
            nf_to = 2 + np.digitize(q2_to, self.area_walls)
        # determine direction and python slice modifier
        if nf_to < nf_from:
            rc = -1
            shift = -3
        else:
            rc = 1
            shift = -2
        # join all necessary points in one list
        boundaries = (
            [q2_from]
            + self.area_walls[nf_from + shift : int(nf_to) + shift : rc]
            + [q2_to]
        )
        segs = [
            PathSegment(boundaries[i], q2, nf_from + i * rc)
            for i, q2 in enumerate(boundaries[1:])
        ]
        return segs

    def nf(self, q2):
        """Find the number of flavors active at the given scale.

        Parameters
        ----------
        q2 : float
            reference scale

        Returns
        -------
        int
            number of active flavors
        """
        ref_idx = np.digitize(q2, self.area_walls)
        return 2 + ref_idx


def is_downward_path(path):
    """Determine if a path is downward.

    Criterias are:

    - in the number of active flavors when the path list contains more than one
      :class:`PathSegment`, note this can be different from each
      :attr:`PathSegment.is_downward_q2`
    - in :math:`Q^2` when just one single :class:`PathSegment` is given

    Parameters
    ----------
    path : list(PathSegment)
        path

    Returns
    -------
    bool
        True for a downward path
    """
    if len(path) == 1:
        return path[0].is_downward_q2
    return path[1].nf < path[0].nf


def flavor_shift(is_downward):
    """Determine the shift to number of light flavors.

    Parameters
    ----------
    is_downward : bool
        True for a downward path

    Returns
    -------
    int
        shift to number of light flavors which can be 3 or 4
    """
    return 4 if is_downward else 3
