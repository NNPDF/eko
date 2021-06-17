# -*- coding: utf-8 -*-
r"""
This module holds the classes that define the |FNS|.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


class PathSegment:
    """
    Oriented path in the threshold landscape.

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
    def tuple(self):
        """Tuple representation suitable for hashing."""
        return (self.q2_from, self.q2_to, self.nf)

    def __repr__(self):
        return f"PathSegment({self.q2_from} -> {self.q2_to}, nf={self.nf})"


class ThresholdsAtlas:
    """
    Holds information about the thresholds any Q2 has to pass in order to get
    there from a given q2_ref.

    Parameters
    ----------
        thresholds: list(float)
            List of q^2 thresholds
        q2_ref: float
            reference scale
    """

    def __init__(self, thresholds, q2_ref=None, nf_ref=None):
        # Initial values
        self.q2_ref = q2_ref
        self.nf_ref = nf_ref
        thresholds = list(thresholds)
        if thresholds != sorted(thresholds):
            raise ValueError("thresholds need to be sorted")
        self.area_walls = [0] + thresholds + [np.inf]
        logger.info("Thresholds: walls = %s", self.area_walls)

    @classmethod
    def ffns(cls, nf, q2_ref=None):
        """
        Create a |FFNS| setup.

        The function creates simply succifienct thresholds at `0` (in the
        beginning), since the number of flavors is determined by counting
        from below.

        Parameters
        ----------
            nf : int
                number of light flavors
            q2_ref : float
                reference scale
        """
        return cls([0] * (nf - 3), q2_ref)

    @classmethod
    def from_dict(cls, theory_card, prefix="k", max_nf_name="MaxNfPdf"):
        r"""
        Create the object from the run card.

        The thresholds are computed by :math:`(m_q \cdot k_q^{Thr})`.

        Parameters
        ----------
            theory_card : dict
                run card with the keys given at the head of the :mod:`module <eko.thresholds>`
            prefix : str
                prefix for the ratio parameters

        Returns
        -------
            ThresholdsAtlas :
                created object
        """

        def thres(pid):
            heavy_flavors = "cbt"
            flavor = heavy_flavors[pid - 4]
            return pow(
                theory_card[f"m{flavor}"] * theory_card[f"{prefix}{flavor}Thr"], 2
            )

        thresholds = [thres(q) for q in range(4, 6 + 1)]
        # cut array = simply reduce some thresholds
        max_nf = theory_card[max_nf_name]
        thresholds = thresholds[: max_nf - 3]
        # preset ref scale
        q2_ref = pow(theory_card["Q0"], 2)
        return cls(thresholds, q2_ref)

    def path(self, q2_to, nf_to=None, q2_from=None, nf_from=None):
        """
        Get path from q2_from to q2_to.

        Parameters
        ----------
            q2_to: float
                target value of q2
            q2_from: float
                starting value of q2

        Returns
        -------
            path: list(PathSegment)
                List of :class:`PathSegment` to go through in order to get from q2_from
                to q2_to.
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
            [q2_from] + self.area_walls[nf_from + shift : nf_to + shift : rc] + [q2_to]
        )
        segs = [
            PathSegment(boundaries[i], q2, nf_from + i * rc)
            for i, q2 in enumerate(boundaries[1:])
        ]
        return segs

    def nf(self, q2):
        """
        Finds the number of flavor active at the given scale.

        Parameters
        ----------
            q2 : float
                reference scale

        Returns
        -------
            nf : int
                number of active flavors
        """
        ref_idx = np.digitize(q2, self.area_walls)
        return 2 + ref_idx
