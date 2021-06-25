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
    def is_backward(self):
        """True if q2_from bigger than q2_to"""
        return self.q2_from > self.q2_to

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
        masses: list(float)
            list of quark masses squared
        q2_ref: float
            reference scale
        nf_ref: int
            number of active flavors at the reference scale
        thresholds_ratios: list(float)
            list of ratios between masses and matching thresholds squared
        max_nf: int
            maximum number of active flavors
    """

    def __init__(
        self, masses, q2_ref=None, nf_ref=None, thresholds_ratios=None, max_nf=None
    ):
        # Initial values
        self.q2_ref = q2_ref
        self.nf_ref = nf_ref

        masses = list(masses)
        if masses != sorted(masses):
            raise ValueError("masses need to be sorted")

        thresholds = self.build_area_walls(masses, thresholds_ratios, max_nf)

        self.area_walls = [0] + thresholds + [np.inf]
        self.thresholds_ratios = thresholds_ratios
        logger.info("Thresholds: walls = %s", self.area_walls)

    def __repr__(self):
        walls = " - ".join(["%.2e" % w for w in self.area_walls])
        return f"ThresholdsAtlas [{walls}], ref={self.q2_ref} @ {self.nf_ref}"

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
        return cls([0] * (nf - 3) + [np.inf] * (6 - nf), q2_ref)

    @staticmethod
    def build_area_walls(masses, thresholds_ratios=None, max_nf=None):
        r"""
        Create the object from the run card.

        The thresholds are computed by :math:`(m_q \cdot k_q^{Thr})`.

        Parameters
        ----------

        Returns
        -------
            list :
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
        heavy_flavors = "cbt"
        masses = [theory_card[f"m{q}"] for q in heavy_flavors]
        thresholds_ratios = [theory_card[f"{prefix}{q}Thr"] for q in heavy_flavors]
        max_nf = theory_card[max_nf_name]
        # preset ref scale
        q2_ref = pow(theory_card["Q0"], 2)
        return cls(
            np.power(masses, 2),
            q2_ref,
            thresholds_ratios=np.power(thresholds_ratios, 2),
            max_nf=max_nf,
        )

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
