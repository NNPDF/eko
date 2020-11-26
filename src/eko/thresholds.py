# -*- coding: utf-8 -*-
r"""
This module holds the classes that define the flavor number schemes (FNS).

Run card parameters:

.. include:: /code/IO-tabs/ThresholdConfig.rst
"""
import logging
import numbers

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
            number of flavours in the area
    """

    def __init__(self, q2_min, q2_max, nf):
        self.q2_min = q2_min
        self.q2_max = q2_max
        self.nf = nf

class PathSegment:
    """
    Oriented path in the threshold area landscape.

    Parameters
    ----------
        q2_from : float
            starting point
        q2_to : float
            final point
        area : Area
            containing area
    """
    def __init__(self, q2_from, q2_to, area):
        self.q2_from = q2_from
        self.q2_to = q2_to
        self.area = area

class ThresholdsConfig:
    """
    The threshold class holds information about the thresholds any
    Q2 has to pass in order to get there from a given q2_ref and scheme.

    Parameters
    ----------
        q2_ref : float
            Reference q^2
        scheme: str
            Choice of scheme
        threshold_list: list
            List of q^2 thresholds should the scheme accept it
        nf: int
            Number of flavors for the FFNS
    """

    def __init__(self, area_walls, q2_ref = None):
        # Initial values
        self.q2_ref = q2_ref
        if area_walls != sorted(area_walls):
            raise ValueError("area walls need to be sorted")
        self.areas = []
        q2_min = 0
        for q2_max in area_walls + [np.inf]:
            nf = nf + 1
            new_area = Area(q2_min, q2_max, nf)
            self.areas.append(new_area)
            q2_min = q2_max

    @classmethod
    def from_dict(cls, theory_card):
        """
        Create the object from the run card.

        Parameters
        ----------
            theory_card : dict
                run card with the keys given at the head of the :mod:`module <eko.thresholds>`

        Returns
        -------
            cls : ThresholdConfig
                created object
        """
        FNS = theory_card["FNS"]
        q2_ref = pow(theory_card["Q0"], 2)
        if FNS == "FFNS":
            return cls(q2_ref, FNS, threshold_list=None, nf=theory_card["NfFF"])
        # the threshold value does not necessarily coincide with the mass
        def thres(pid):
            heavy_flavors = "cbt"
            flavor = heavy_flavors[pid - 4]
            return pow(theory_card[f"m{flavor}"] * theory_card[f"k{flavor}Thr"], 2)
        if FNS in ["FONLL-A"]:
            nf = theory_card["NfFF"]
            if nf <= 3:
                raise ValueError("NfFF should point to the heavy quark! i.e. NfFF>3")
            threshold_list = [thres(nf)]
            return cls(q2_ref, FNS, threshold_list=threshold_list, nf=nf-1)
        # setup VFNS
        threshold_list = [thres(q) for q in range(4, 6 + 1)]
        return cls(q2_ref, FNS, threshold_list=threshold_list)

    @property
    def nf_ref(self):
        """ Number of flavours in the reference area """
        return self._areas[self._area_ref].nf

    def get_path_from_q2_ref(self, q2):
        """
        Get the Area path from q2_ref to q2.

        Parameters
        ----------
            q2: float
                Target value of q2

        Returns
        -------
            area_path: list
                List of Areas to go through in order to get from q2_ref
                to q2. The first one is the one containg q2_ref while the
                last one contains q2
        """
        current_area = self.get_areas_idx(q2)[0]
        if current_area < self._area_ref:
            rc = -1
        else:
            rc = 1
        area_path = [
            self._areas[i] for i in range(self._area_ref, current_area + rc, rc)
        ]
        return area_path

    def get_areas_idx(self, q2arr):
        """
        Returns the index of the area in which each value of q2arr falls.

        Parameters
        ----------
            q2arr: np.array
                array of values of q2

        Returns
        -------
            areas_idx: list
                list with the indices of the corresponding areas for q2arr
        """
        # Ensure q2arr is an array
        if isinstance(q2arr, numbers.Number):
            q2arr = np.array([q2arr])
        # Check in which area is every q2
        areas_idx = np.digitize(q2arr, self._area_walls)
        return areas_idx

    def get_areas(self, q2arr):
        """
        Returns the Areas in which each value of q2arr falls

        Parameters
        ----------
            q2arr: np.array
                array of values of q2

        Returns
        -------
            areas: list
                list with the areas for q2arr
        """
        idx = self.get_areas_idx(q2arr)
        area_list = np.array(self._areas)[idx]
        return list(area_list)
