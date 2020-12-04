# -*- coding: utf-8 -*-
r"""
This module holds the classes that define the |FNS|.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


class Area:
    """
    Sets up a single threshold area with a fixed configuration.

    Parameters
    ----------
        q2_min : float
            lower bound of the area
        q2_max : float
            upper bound of the area
        nf : float
            number of flavors in the area
    """

    def __init__(self, q2_min, q2_max, nf):
        self.q2_min = q2_min
        self.q2_max = q2_max
        self.nf = nf

    def __repr__(self):
        return f"Area([{self.q2_min},{self.q2_max}], nf={self.nf})"


class PathSegment:
    """
    Oriented path in the threshold area landscape.

    Parameters
    ----------
        q2_from : float
            starting point
        q2_to : float
            final point
        area : eko.thresholds.Area
            containing area
    """

    def __init__(self, q2_from, q2_to, area):
        self.q2_from = q2_from
        self.q2_to = q2_to
        self._area = area

    @property
    def nf(self):
        return self._area.nf

    @property
    def tuple(self):
        return (self.q2_from, self.q2_to)

    def __repr__(self):
        return f"PathSegment({self.q2_from} -> {self.q2_to}, nf={self.nf})"

    @classmethod
    def intersect(cls, q2_from, q2_to, area):
        if q2_from < q2_to:
            return cls(max(q2_from, area.q2_min), min(q2_to, area.q2_max), area)
        else:
            return cls(min(q2_from, area.q2_max), max(q2_to, area.q2_min), area)


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

    def __init__(self, thresholds, q2_ref=None):
        # Initial values
        self.q2_ref = q2_ref
        thresholds = list(thresholds)
        if thresholds != sorted(thresholds):
            raise ValueError("thresholds need to be sorted")
        self.areas = []
        self.area_walls = [0] + thresholds + [np.inf]
        q2_min = 0
        nf = 3
        for q2_max in self.area_walls[1:]:
            new_area = Area(q2_min, q2_max, nf)
            self.areas.append(new_area)
            nf = nf + 1
            q2_min = q2_max

    @classmethod
    def ffns(cls, nf, q2_ref=None):
        """
        Create a |FFNS| setup.

        Parameters
        ----------
            nf : int
                number of light flavors
            q2_ref : float
                reference scale
        """
        return cls([0] * (nf - 3) + [np.inf] * (6 - nf), q2_ref)

    @classmethod
    def from_dict(cls, theory_card, prefix="k"):
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
            cls : ThresholdsAtlas
                created object
        """

        def thres(pid):
            heavy_flavors = "cbt"
            flavor = heavy_flavors[pid - 4]
            return pow(
                theory_card[f"m{flavor}"] * theory_card[f"{prefix}{flavor}Thr"], 2
            )

        thresholds = [thres(q) for q in range(4, 6 + 1)]
        # preset ref scale
        q2_ref = pow(theory_card["Q0"], 2)
        return cls(thresholds, q2_ref)

    def path(self, q2_to, q2_from=None):
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
                List of PathSegment to go through in order to get from q2_from
                to q2_to.
        """
        if q2_from is None:
            q2_from = self.q2_ref

        ref_idx = np.digitize(q2_from, self.area_walls)
        target_idx = np.digitize(q2_to, self.area_walls)
        if q2_to < q2_from:
            rc = -1
        else:
            rc = 1
        path = [
            PathSegment.intersect(q2_from, q2_to, self.areas[i - 1])
            for i in range(ref_idx, target_idx + rc, rc)
        ]
        return list(filter(lambda s: not np.allclose(s.q2_from, s.q2_to), path))

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
        return self.areas[ref_idx - 1].nf
